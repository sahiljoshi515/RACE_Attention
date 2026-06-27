"""Long-context + synthetic-retrieval + curriculum data for RACE-hybrid distillation.

Distillation is LABEL-FREE: the frozen teacher's logits are the supervision target, so
every generator here only needs to emit realistic INPUT token sequences `[B, seq_length]`
(exactly seq_length; pad with eos, never produce labels). Three data kinds:

  * `packed_long_batches` — genuinely-long natural docs (PG19 / FineWeb-edu) packed to
    seq_length, so a long block is real continuous text (teaches long-range LM), not 200
    EOS-glued fragments the way packing short FineWeb docs would be.
  * `synthetic_ruler_batch` — RULER-style retrieval tasks (NIAH variants, variable
    tracking, common/frequent-words, QA) matching the phrasings eval_ruler.py scores
    (QUESTION_ANCHORS) and the SAME chat templating as eval_ruler.build_prompt_ids, in two
    formats: query-agnostic (question only at the end, mirrors att-hub ruler-32k) and
    query-in-context (question also embedded in context, mirrors tonychenxyz ruler-64k).
  * `mixed_batches` / `make_curriculum` — a 70/30 LM:retrieval mix and a length curriculum
    (ramp 4K->...->64K) so the RACE recurrent state + RoPE phases are exercised at the
    eval regime.

Assembly note: needles/filler are composed as TEXT and the assembled context+question is
tokenized ONCE per sample (cheap relative to a 3B fwd/bwd), then fixed to exactly
seq_length at the token level — keeping the chat-template prefix (head) and the
question+assistant-header tail intact, trimming/padding only the filler in between. The
filler TEXT pool is built ONCE and string-sliced per sample (no per-sample re-tokenization
of the huge filler).
"""
import random

import torch

from data_fineweb import get_tokenizer  # noqa: F401 (re-exported for callers)

# Question phrasings copied to match eval_ruler.QUESTION_ANCHORS exactly (train==eval format).
ANCHOR_NUMBER = "What is the special magic number for"
ANCHOR_UUID = "What is the special magic uuid for"
ANCHOR_MULTI = "What are all the special magic numbers for"
ANCHOR_VT = "Question: Find all variables that are assigned the value"
ANCHOR_CWE = "Question: What are the 10 most common words"
ANCHOR_FWE = "Question: Do not provide any explanation. What are the 3 most frequently appeared words"
ANCHOR_QA = "Answer the question based on the given documents"

SYNTH_TASKS = ("niah_single", "niah_single_uuid", "niah_multikey", "niah_multivalue",
               "niah_multiquery", "variable_tracking", "common_words", "frequent_words", "qa")

_FILLER_CACHE = {}  # (source, seed) -> big filler text string


def _open_long_source(source="fineweb_local", seed=0):
    """Open a long-document TEXT dataset for filler/LM packing -> (ds, field).
    'fineweb_local' streams the pre-staged local FineWeb parquet shards (network-free, same
    data as data_fineweb) -- the reliable default under datasets 4.x, where script-based
    'pg19' no longer loads ('Dataset scripts are no longer supported, but found pg19.py')."""
    from datasets import load_dataset
    import glob
    if source == "fineweb_local":
        shards = sorted(glob.glob("/scratch/sj157/RACE_Attention/data/fineweb/**/*.parquet",
                                  recursive=True))
        if shards:
            return load_dataset("parquet", data_files=shards, split="train",
                                streaming=True), "text"
        source = "fineweb_edu"   # no local shards -> hub fallback (parquet-backed, no script)
    name, field = {"pg19": ("deepmind/pg19", "text"),
                   "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "text")}.get(
                       source, ("HuggingFaceFW/fineweb-edu", "text"))
    return load_dataset(name, split="train", streaming=True), field


# --------------------------------------------------------------------------- filler pool
def _build_filler_text(source="fineweb_local", seed=0, min_chars=3_000_000):
    """Accumulate a big natural-language string once (streamed), cached. Falls back to a
    deterministic pseudo-corpus when the dataset/network is unavailable (offline smoke)."""
    key = (source, seed)
    if key in _FILLER_CACHE:
        return _FILLER_CACHE[key]
    text = ""
    try:
        ds, field = _open_long_source(source, seed)
        if seed:
            ds = ds.shuffle(seed=seed, buffer_size=1000)
        parts = []
        got = 0
        for ex in ds:
            t = ex.get(field) or ""
            if len(t) < 2000:           # skip short docs; we want genuinely-long text
                continue
            parts.append(t)
            got += len(t)
            if got >= min_chars:
                break
        text = "\n\n".join(parts)
    except Exception as e:               # offline / dataset error -> synthetic filler
        rng = random.Random(seed or 1)
        words = ["the", "of", "and", "to", "in", "a", "that", "was", "his", "he", "with",
                 "it", "as", "for", "her", "had", "is", "at", "but", "on", "not", "they",
                 "river", "mountain", "evening", "letter", "garden", "window", "memory"]
        buf = []
        while sum(len(w) + 1 for w in buf) < min_chars:
            buf.append(rng.choice(words))
            if rng.random() < 0.05:
                buf.append(".\n")
        text = " ".join(buf)
        print(f"[data_long] WARN: filler dataset unavailable ({type(e).__name__}); "
              f"using synthetic filler ({len(text)} chars)")
    _FILLER_CACHE[key] = text
    return text


# --------------------------------------------------------------------------- token helpers
def _fit(ids, seq_length, eos, keep_head=44):
    """Force `ids` to exactly seq_length: trim filler from just after the chat-template
    head (preserving both the system/user header AND the question+assistant-header tail),
    or left-pad with eos. Padding/trimming the filler is benign — the teacher sees the
    identical input, so it just shapes the supervision context."""
    if len(ids) > seq_length:
        over = len(ids) - seq_length
        head = min(keep_head, len(ids))
        ids = ids[:head] + ids[head + over:]
    if len(ids) < seq_length:
        ids = [eos] * (seq_length - len(ids)) + ids
    return ids[:seq_length]


def _rand_word(rng, lo=5, hi=7):
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz") for _ in range(rng.randint(lo, hi)))


def _rand_number(rng):
    return str(rng.randint(1_000_000, 9_999_999))


def _rand_uuid(rng):
    h = "0123456789abcdef"
    g = lambda n: "".join(rng.choice(h) for _ in range(n))
    return f"{g(8)}-{g(4)}-{g(4)}-{g(4)}-{g(12)}"


# --------------------------------------------------------------------------- task builders
def _build_task(task, rng, n_distractors=0):
    """Return (needles:list[str], question:str, answer_prefix:str) for a synthetic task.
    The answer itself is irrelevant (label-free KD); only realistic structure matters.
    `n_distractors` injects SAME-pattern decoys so the task matches RULER's hard variants
    (distractor-heavy niah / adversarial variable-tracking) our hybrid currently fails."""
    if task in ("niah_single", "niah_single_uuid"):
        key = _rand_word(rng)
        uuid = task.endswith("uuid")
        val = _rand_uuid(rng) if uuid else _rand_number(rng)
        word = "uuid" if uuid else "number"
        anchor = ANCHOR_UUID if uuid else ANCHOR_NUMBER
        needles = [f"One of the special magic {word}s is hidden in the text. "
                   f"The special magic {word} for {key} is {val}."]
        # Distractor needles: identical phrase pattern, DIFFERENT keys -> the model must
        # disambiguate by the queried key instead of pattern-matching/looping on the phrase
        # (mirrors RULER niah_single_1, the variant our hybrid fails by repetition-looping).
        for _ in range(n_distractors):
            dk = _rand_word(rng)
            dv = _rand_uuid(rng) if uuid else _rand_number(rng)
            needles.append(f"The special magic {word} for {dk} is {dv}.")
        return needles, f" {anchor} {key}? ", f" The special magic {word} for {key} is"

    if task == "niah_multikey":
        n = rng.randint(3, 6)
        pairs = [(_rand_word(rng), _rand_number(rng)) for _ in range(n)]
        needles = [f"The special magic number for {k} is {v}." for k, v in pairs]
        k, _ = rng.choice(pairs)
        return needles, f" {ANCHOR_NUMBER} {k}? ", f" The special magic number for {k} is"

    if task == "niah_multivalue":
        key = _rand_word(rng)
        vals = [_rand_number(rng) for _ in range(rng.randint(2, 4))]
        needles = [f"The special magic number for {key} is {v}." for v in vals]
        return needles, f" {ANCHOR_MULTI} {key}? ", f" The special magic numbers for {key} are"

    if task == "niah_multiquery":
        keys = [_rand_word(rng) for _ in range(rng.randint(2, 4))]
        needles = [f"The special magic number for {k} is {_rand_number(rng)}." for k in keys]
        return needles, f" {ANCHOR_MULTI} {', '.join(keys)}? ", " The special magic numbers are"

    if task == "variable_tracking":
        root = _rand_number(rng)
        chain = [_rand_word(rng).upper() for _ in range(rng.randint(3, 5))]
        needles = [f"VAR {chain[0]} = {root}."]
        needles += [f"VAR {chain[i]} = {chain[i-1]}." for i in range(1, len(chain))]
        # Adversarial decoys: fake VARs holding OTHER values, so the chain can't be solved by
        # grabbing any assignment (mirrors RULER vt's distractor assignments our hybrid fails).
        for _ in range(min(n_distractors, 8)):
            needles.append(f"VAR {_rand_word(rng).upper()} = {_rand_number(rng)}.")
        return needles, f" {ANCHOR_VT} {root}. ", " The variables are"

    if task in ("common_words", "frequent_words"):
        vocab = [_rand_word(rng) for _ in range(rng.randint(20, 30))]
        # a few words repeated many times so there IS a most-common set
        hot = rng.sample(vocab, 10)
        seq = []
        for w in vocab:
            seq += [w] * rng.randint(1, 3)
        for w in hot:
            seq += [w] * rng.randint(5, 12)
        rng.shuffle(seq)
        # Embed the words in light prose (RULER cwe has words within sentences, not a bare list).
        sents = [f"The text mentions {', '.join(seq[i:i+8])}." for i in range(0, len(seq), 8)]
        needles = [" ".join(sents)]
        anchor = ANCHOR_CWE if task == "common_words" else ANCHOR_FWE
        return needles, f" {anchor} in the list above? ", " The most common words are"

    # qa: a short pseudo-document + a question about it
    subj = _rand_word(rng)
    fact = _rand_number(rng)
    needles = [f"Document: The {subj} project was completed in {fact} and was widely praised."]
    return needles, f" {ANCHOR_QA}. Question: In what year was the {subj} project completed? ", " Answer:"


def _make_sample_ids(tok, task, fmt, seq_length, rng, filler_text, n_distractors=0):
    """Assemble ONE [seq_length] sample: needles embedded in a filler slice, chat-templated
    like eval_ruler.build_prompt_ids, tokenized once, fixed to exact length."""
    eos = tok.eos_token_id
    needles, question, answer_prefix = _build_task(task, rng, n_distractors=n_distractors)

    # Take a filler slice sized to land near seq_length tokens (~4.5 chars/token), with
    # margin for needles/question; exact length is fixed afterward at the token level.
    approx_chars = int(seq_length * 4.5)
    if len(filler_text) > approx_chars + 1:
        start = rng.randint(0, len(filler_text) - approx_chars - 1)
        filler = filler_text[start:start + approx_chars]
    else:
        filler = (filler_text * (approx_chars // max(1, len(filler_text)) + 1))[:approx_chars]

    # Scatter the needles (and, for query_in_context, the question text) into the filler.
    inserts = list(needles)
    if fmt == "query_in_context":
        inserts.append(question.strip())
    words = filler.split(" ")
    for piece in inserts:
        pos = rng.randint(0, len(words))
        words.insert(pos, " " + piece + " ")
    context = " ".join(words)

    # SAME templating path as eval_ruler.build_prompt_ids (context+question -> chat -> +prefix).
    msg = [{"role": "user", "content": context + question}]
    s = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    s = s + answer_prefix
    ids = tok(s, add_special_tokens=False).input_ids
    return _fit(ids, seq_length, eos)


# --------------------------------------------------------------------------- public generators
def packed_long_batches(tokenizer, seq_length=4096, batch_size=2, skip_batches=0,
                        max_batches=None, seed=0, source="pg19", min_doc_tokens=None):
    """Yield [B,seq_length] long tensors packed from genuinely-long docs (PG19/FineWeb-edu).
    Mirrors data_fineweb.packed_batches but filters to docs with >= (min_doc_tokens or
    seq_length) tokens so a packed block is real continuous text."""
    ds, field = _open_long_source(source, seed)
    if seed:
        ds = ds.shuffle(seed=seed, buffer_size=1000)
    eos = tokenizer.eos_token_id
    # FineWeb web docs are mostly < seq_length tokens; a hard >=seq_length filter would starve
    # at 4K/16K. For local FineWeb pack short docs contiguously (like data_fineweb) with a small
    # floor; keep the genuine-long-doc filter only for sources that actually have long docs.
    min_tok = min_doc_tokens or (256 if source == "fineweb_local" else seq_length)
    tok_buf, batch_buf, produced = [], [], 0
    for ex in ds:
        ids = tokenizer.encode(ex.get(field) or "", add_special_tokens=False)
        if len(ids) < min_tok:                  # keep long docs only
            continue
        ids.append(eos)
        tok_buf.extend(ids)
        while len(tok_buf) >= seq_length:
            batch_buf.append(torch.tensor(tok_buf[:seq_length], dtype=torch.long))
            tok_buf = tok_buf[seq_length:]
            if len(batch_buf) == batch_size:
                if produced < skip_batches:
                    produced += 1; batch_buf = []; continue
                yield torch.stack(batch_buf)
                batch_buf = []; produced += 1
                if max_batches is not None and (produced - skip_batches) >= max_batches:
                    return


def synthetic_ruler_batch(tokenizer, seq_length=4096, batch_size=2, seed=0,
                          task=None, fmt="query_agnostic", filler_pool=None,
                          max_batches=None, filler_source="pg19",
                          task_weights=None, n_distractors=0):
    """Yield [B,seq_length] long tensors of RULER-style retrieval tasks (label-free inputs).
    task=None -> random per sample (weighted by task_weights if given); fmt in
    {query_agnostic, query_in_context, None(random)}. filler_pool: optional prebuilt filler TEXT.
    task_weights: {task: weight} to oversample hard tasks; n_distractors: same-pattern decoys
    injected into niah/vt to match RULER's hard variants."""
    rng = random.Random(seed)
    filler_text = filler_pool if isinstance(filler_pool, str) else \
        _build_filler_text(source=filler_source, seed=seed)
    weights = [task_weights.get(t, 1.0) for t in SYNTH_TASKS] if task_weights else None
    produced = 0
    while True:
        rows = []
        for _ in range(batch_size):
            tk = task or (rng.choices(SYNTH_TASKS, weights=weights, k=1)[0]
                          if weights else rng.choice(SYNTH_TASKS))
            fm = fmt or rng.choice(("query_agnostic", "query_in_context"))
            rows.append(_make_sample_ids(tokenizer, tk, fm, seq_length, rng, filler_text,
                                         n_distractors=n_distractors))
        yield torch.tensor(rows, dtype=torch.long)
        produced += 1
        if max_batches is not None and produced >= max_batches:
            return


def mixed_batches(tokenizer, seq_length=4096, batch_size=2, lm_frac=0.7, seed=0,
                  skip_batches=0, max_batches=None, source="pg19",
                  task_weights=None, n_distractors=0):
    """70/30 (default) LM:retrieval mix. Per batch, draw LM from packed_long_batches with
    prob lm_frac, else a synthetic retrieval task (weighted by task_weights, with n_distractors
    decoys for the hard niah/vt variants)."""
    rng = random.Random(seed)
    lm_gen = packed_long_batches(tokenizer, seq_length, batch_size, skip_batches=skip_batches,
                                 seed=seed, source=source)
    rt_gen = synthetic_ruler_batch(tokenizer, seq_length, batch_size, seed=seed + 1,
                                   task=None, fmt=None, filler_source=source,
                                   task_weights=task_weights, n_distractors=n_distractors)
    produced = 0
    while True:
        gen = lm_gen if rng.random() < lm_frac else rt_gen
        try:
            yield next(gen)
        except StopIteration:                   # LM stream exhausted -> fall back to retrieval
            yield next(rt_gen)
        produced += 1
        if max_batches is not None and produced >= max_batches:
            return


class CurriculumLoader:
    """Length curriculum: `batch_for_step(step)` returns a [B,T] batch at the seq_length for
    the current step, rebuilding the underlying 70/30 mixer when a stage boundary is crossed.
    `current_seqlen` is exposed for logging. B per stage comes from batch_size_fn(seq_length)
    (longer seq -> smaller B). Schedule = sorted list of (start_step, seq_length)."""

    def __init__(self, tokenizer, schedule, batch_size_fn, lm_frac=0.7, seed=0, source="pg19",
                 task_weights=None, n_distractors=0):
        self.tok = tokenizer
        self.schedule = sorted(schedule)
        self.batch_size_fn = batch_size_fn
        self.lm_frac = lm_frac
        self.seed = seed
        self.source = source
        self.task_weights = task_weights
        self.n_distractors = n_distractors
        self.current_seqlen = None
        self._gen = None

    def _seqlen_for(self, step):
        sl = self.schedule[0][1]
        for start, length in self.schedule:
            if step >= start:
                sl = length
        return sl

    def batch_for_step(self, step):
        sl = self._seqlen_for(step)
        if sl != self.current_seqlen:           # crossed a stage boundary -> rebuild mixer
            self.current_seqlen = sl
            bsz = self.batch_size_fn(sl)
            self._gen = mixed_batches(self.tok, seq_length=sl, batch_size=bsz,
                                      lm_frac=self.lm_frac, seed=self.seed + sl,
                                      source=self.source, task_weights=self.task_weights,
                                      n_distractors=self.n_distractors)
        return next(self._gen)


def make_curriculum(tokenizer, schedule, batch_size_fn, lm_frac=0.7, seed=0, source="pg19",
                    task_weights=None, n_distractors=0):
    """Factory for CurriculumLoader (the form the trainer uses: loader.batch_for_step(step))."""
    return CurriculumLoader(tokenizer, schedule, batch_size_fn, lm_frac=lm_frac,
                            seed=seed, source=source, task_weights=task_weights,
                            n_distractors=n_distractors)


# Generator form of the curriculum, kept for API completeness (send the current step in).
def curriculum_train_gen(tokenizer, schedule, batch_size_fn, lm_frac=0.7, seed=0, source="pg19"):
    """Generator variant: `g.send(step)` (after priming with next(g)) yields the batch for
    that step. The CurriculumLoader.batch_for_step form above is what the trainer uses."""
    loader = CurriculumLoader(tokenizer, schedule, batch_size_fn, lm_frac=lm_frac,
                              seed=seed, source=source)
    step = 0
    while True:
        recv = yield loader.batch_for_step(step)
        step = recv if recv is not None else step + 1


def eval_probe_batch(tokenizer, seq_length=4096, seed=0, source="pg19"):
    """A single [1,seq_length] held-out long-doc batch for the trainer's ppl probe."""
    gen = packed_long_batches(tokenizer, seq_length, batch_size=1, max_batches=1,
                              seed=seed, source=source)
    return next(gen)


if __name__ == "__main__":
    SL, B = 512, 2
    tok = get_tokenizer()
    eos = tok.eos_token_id

    def chk(name, t):
        assert t.dtype == torch.long and tuple(t.shape) == (t.shape[0], SL), \
            f"{name}: bad shape/dtype {t.shape} {t.dtype}"
        snip = tok.decode(t[0][:40]).replace("\n", " ")
        print(f"{name:34s} {tuple(t.shape)} {t.dtype} | {snip[:90]}")

    # synthetic path needs only the tokenizer (works offline) — validate every task+format.
    for task in SYNTH_TASKS:
        for fmt in ("query_agnostic", "query_in_context"):
            g = synthetic_ruler_batch(tok, SL, B, seed=1, task=task, fmt=fmt,
                                       filler_source="pg19", max_batches=1)
            chk(f"synth:{task}/{fmt[:5]}", next(g))

    # curriculum (uses mixer -> may hit the dataset; guarded)
    try:
        loader = make_curriculum(tok, [(0, SL), (1, SL)], batch_size_fn=lambda s: B, seed=2)
        chk("curriculum@step0", loader.batch_for_step(0))
        chk("curriculum@step1", loader.batch_for_step(1))
    except Exception as e:
        print(f"curriculum/mixer needs dataset access ({type(e).__name__}: {e})")

    # LM + eval-probe paths need the dataset (compute nodes have internet); guard for login.
    try:
        chk("packed_long", next(packed_long_batches(tok, SL, B, max_batches=1, seed=3)))
        chk("mixed", next(mixed_batches(tok, SL, B, lm_frac=0.5, seed=4, max_batches=1)))
        chk("eval_probe", eval_probe_batch(tok, SL, seed=5))
    except Exception as e:
        print(f"long-doc paths need dataset access ({type(e).__name__}: {e})")

    print("data_long smoke OK")
