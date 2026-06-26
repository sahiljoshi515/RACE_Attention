"""FineWeb streaming + packing for the distillation pilot.

Streams HuggingFaceFW/fineweb, tokenizes with the Llama-3.2 tokenizer
(add_special_tokens=False, EOS between documents), packs into contiguous
seq_length blocks, and yields [B, seq_length] long batches. A fixed held-out
eval set is taken from the front of the stream and skipped for training.

Network resilience
------------------
The HuggingFace datasets streaming API intermittently raises HTTP 504 /
connection / read-timeout errors mid-stream, which would otherwise kill a long
training run.  ``packed_batches`` wraps the iteration so that on a *network*
error it logs a warning, backs off (exponential, capped), RE-CREATES the
dataset stream (re-``load_dataset`` + re-``shuffle`` with the SAME seed) and
continues yielding fresh packed batches.

NOTE ON DATA ORDER: because we re-create + re-shuffle the stream with the same
seed, after a reconnect we begin re-reading documents from the top of the
(reshuffled) stream.  Training only needs fresh tokens, so re-yielding from a
reshuffled stream is acceptable; we do NOT guarantee exactly-once / no-repeat
document order across reconnects.  Batch *counting* (skip_batches / max_batches)
is preserved exactly: a ``produced`` counter (and the partial token/batch
buffers) persists across reconnects, so already-served batches are never
re-skipped or re-counted.

Only network/streaming exceptions are caught; genuine bugs (KeyError, etc.)
propagate, and a normal end-of-stream (StopIteration) ends the generator
normally.  A consecutive-failure cap guards against an infinite reconnect storm;
the cap counts CONSECUTIVE failures only -- any successful batch resets it, so a
handful of 504s spread over a long run never accumulate into a false give-up.
"""
import os
import time
import logging

# Backstop: bump the hub download timeout before anything imports/uses it.
# (Honored by huggingface_hub; a longer timeout reduces spurious ReadTimeouts.)
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

import torch
import datasets
from datasets import load_dataset
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# Backoff policy for streaming reconnects.
_MAX_CONSECUTIVE_FAILURES = 8   # cap before re-raising (guards reconnect storm)
_BACKOFF_BASE = 2.0             # seconds; doubled each consecutive failure
_BACKOFF_CAP = 60.0            # seconds
_DOWNLOAD_MAX_RETRIES = 5      # passed to datasets.DownloadConfig if supported


# Exception types that we treat as transient network/streaming failures.
# Built dynamically so a missing optional dependency never breaks import.
def _network_error_types():
    types = []
    # Standard library / sockets.
    try:
        import socket
        types.append(socket.timeout)
        types.append(socket.gaierror)
    except Exception:
        pass
    types.append(ConnectionError)   # builtin; parent of many net errors
    types.append(TimeoutError)      # builtin
    # requests
    try:
        import requests
        types.append(requests.exceptions.RequestException)
    except Exception:
        pass
    # urllib3
    try:
        import urllib3
        types.append(urllib3.exceptions.HTTPError)
    except Exception:
        pass
    # huggingface_hub
    try:
        import huggingface_hub.utils as _hf_utils
        for name in ("HfHubHTTPError", "OfflineModeIsEnabled"):
            t = getattr(_hf_utils, name, None)
            if isinstance(t, type) and issubclass(t, BaseException):
                types.append(t)
    except Exception:
        pass
    # aiohttp (datasets streaming uses fsspec/aiohttp under the hood)
    try:
        import aiohttp
        types.append(aiohttp.ClientError)
        # ServerTimeoutError / asyncio timeout surface as builtin TimeoutError
        # subclasses in most versions, already covered above.
    except Exception:
        pass
    return tuple(types)


_NETWORK_ERRORS = _network_error_types()


def _is_network_error(exc):
    """True if exc (or any chained cause/context) looks like a transient
    network/streaming failure. Walks __cause__/__context__ because datasets
    often wraps the underlying aiohttp/requests error."""
    seen = set()
    cur = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, _NETWORK_ERRORS):
            return True
        # Heuristic fallback for HTTP status text (e.g. "504") not covered by
        # a recognized type, while still NOT matching plain logic bugs.
        msg = str(cur)
        if any(code in msg for code in ("504", "503", "502", "500")) and \
           any(w in msg for w in ("Gateway", "gateway", "Server Error",
                                  "server error", "Timeout", "timeout",
                                  "Connection", "connection")):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def get_tokenizer(model=MODEL):
    return AutoTokenizer.from_pretrained(model)


# Local pre-staged FineWeb shards (see stage_fineweb.py). When present we read these
# directly with NO hub access — the hub's dataset tree-listing API 504s/stalls at job
# start even for sequential (unshuffled) streaming, which left GPUs idle and got jobs
# idle-reaped. Local parquet streaming is network-free and reaper-safe.
_LOCAL_FINEWEB = "/scratch/sj157/RACE_Attention/data/fineweb"


def _local_fineweb_shards():
    import glob
    if os.environ.get("FINEWEB_NO_LOCAL"):     # escape hatch to force hub streaming
        return []
    return sorted(glob.glob(os.path.join(_LOCAL_FINEWEB, "**", "*.parquet"), recursive=True))


def _open_stream(seed):
    """(Re)create the FineWeb streaming dataset, with robust download config
    and the same shuffle seed so reconnects are deterministic. Prefers locally
    pre-staged parquet shards (no network) when available."""
    local = _local_fineweb_shards()
    if local:
        ds = load_dataset("parquet", data_files=local, split="train", streaming=True)
    else:
        kwargs = dict(split="train", streaming=True)
        # Pass a DownloadConfig(max_retries=...) when supported by this datasets ver.
        try:
            dl_cfg = datasets.DownloadConfig(max_retries=_DOWNLOAD_MAX_RETRIES)
            kwargs["download_config"] = dl_cfg
        except Exception:
            # Older/newer datasets without that arg: silently fall back.
            pass
        try:
            ds = load_dataset("HuggingFaceFW/fineweb", **kwargs)
        except TypeError:
            # download_config not accepted by load_dataset on this version.
            kwargs.pop("download_config", None)
            ds = load_dataset("HuggingFaceFW/fineweb", **kwargs)
    # Streaming shuffle is OPT-IN (env FINEWEB_SHUFFLE_BUFFER>0). It was the hang cause:
    # ds.shuffle(seed) enumerates FineWeb's full shard tree (the recursive .../tree/ API
    # call that 504'd / silently stalled), independent of buffer_size. Sequential reads are
    # fast (~200 docs/s) and, for the ablation, IDENTICAL across cells -> cleaner comparison.
    sbuf = int(os.environ.get("FINEWEB_SHUFFLE_BUFFER", "0"))
    if seed and sbuf > 0:
        ds = ds.shuffle(seed=seed, buffer_size=sbuf)
    return ds


def packed_batches(tokenizer, seq_length=4096, batch_size=8, skip_batches=0,
                   max_batches=None, seed=0):
    """Yield (input_ids[B,T]) long tensors of packed FineWeb tokens.

    Resilient to transient HF streaming network errors: on such an error the
    stream is re-created (re-shuffled with the same seed) and iteration resumes
    while preserving the persistent ``produced`` counter and partial buffers, so
    skip_batches / max_batches accounting stays exact across reconnects.
    """
    eos = tokenizer.eos_token_id
    # State that MUST persist across reconnects.
    tok_buf, batch_buf, produced = [], [], 0
    consecutive_failures = 0

    while True:
        ds = _open_stream(seed)
        try:
            for ex in ds:
                # Tokenization / dict access are NOT network ops: any error here
                # (e.g. KeyError) is a real bug and must propagate.
                ids = tokenizer.encode(ex["text"], add_special_tokens=False)
                ids.append(eos)
                tok_buf.extend(ids)
                while len(tok_buf) >= seq_length:
                    batch_buf.append(torch.tensor(tok_buf[:seq_length],
                                                  dtype=torch.long))
                    tok_buf = tok_buf[seq_length:]
                    if len(batch_buf) == batch_size:
                        if produced < skip_batches:    # skip held-out region
                            produced += 1
                            batch_buf = []
                            continue
                        yield torch.stack(batch_buf)
                        batch_buf = []
                        produced += 1
                        # A successful yield = real forward progress: reset the
                        # consecutive-failure counter so 504s spread over a long
                        # run don't accumulate into a false give-up.
                        consecutive_failures = 0
                        if max_batches is not None and \
                           (produced - skip_batches) >= max_batches:
                            return
            # Stream ended normally (StopIteration on the for-loop). FineWeb is
            # effectively unbounded, but if it ever ends we stop cleanly.
            return
        except GeneratorExit:
            # Consumer stopped iterating / generator closed: do not retry.
            raise
        except BaseException as exc:  # noqa: BLE001 - re-raised unless network
            if not _is_network_error(exc):
                # Real bug (KeyError, OOM, etc.): never swallow.
                raise
            consecutive_failures += 1
            if consecutive_failures > _MAX_CONSECUTIVE_FAILURES:
                logger.error(
                    "FineWeb stream failed %d consecutive times; giving up.",
                    consecutive_failures - 1)
                raise
            sleep_s = min(_BACKOFF_CAP,
                          _BACKOFF_BASE * (2 ** (consecutive_failures - 1)))
            logger.warning(
                "FineWeb streaming network error (attempt %d/%d): %s: %s. "
                "Reconnecting in %.1fs (produced=%d batches so far).",
                consecutive_failures, _MAX_CONSECUTIVE_FAILURES,
                type(exc).__name__, exc, sleep_s, produced)
            time.sleep(sleep_s)
            # Loop: re-open the (reshuffled, same-seed) stream and resume.
            # tok_buf / batch_buf / produced persist intentionally.
            continue


def make_eval_and_train(tokenizer, seq_length=4096, batch_size=8,
                        num_eval_batches=2, max_train_batches=100, seed=0):
    """Returns (list of eval batches, train generator) that are disjoint."""
    eval_batches = list(packed_batches(tokenizer, seq_length, batch_size,
                                       skip_batches=0, max_batches=num_eval_batches,
                                       seed=seed))
    train_gen = packed_batches(tokenizer, seq_length, batch_size,
                               skip_batches=num_eval_batches,
                               max_batches=max_train_batches, seed=seed)
    return eval_batches, train_gen


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tok = get_tokenizer()
    eb, tg = make_eval_and_train(tok, seq_length=256, batch_size=2,
                                 num_eval_batches=1, max_train_batches=2)
    print("eval batches:", len(eb), eb[0].shape if eb else None)
    b = next(tg)
    print("train batch:", b.shape, "first tokens:", b[0, :8].tolist())
