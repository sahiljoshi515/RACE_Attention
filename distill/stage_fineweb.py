"""Pre-stage FineWeb parquet shards to local /scratch so training never depends on the
flaky HF streaming tree-listing API at job start (which 504s and leaves GPUs idle ->
the cluster idle-reaper SIGTERMs the job). Uses direct hf_hub_download of the small
sample/10BT subset shards, which bypasses the dataset tree API. Run once.
"""
import os
import sys
import time

DEST = "/scratch/sj157/RACE_Attention/data/fineweb"
REPO = "HuggingFaceFW/fineweb"
# sample/10BT = the curated 10B-token sample; shards are a few hundred MB each (plenty:
# the ablation needs ~0.1B tokens, Phase 1 ~1-2B). Grab a few shards for headroom.
N_SHARDS = int(sys.argv[1]) if len(sys.argv) > 1 else 3


def main():
    os.makedirs(DEST, exist_ok=True)
    from huggingface_hub import hf_hub_download
    got = []
    for i in range(N_SHARDS):
        fn = f"sample/10BT/{i:03d}_00000.parquet"
        for attempt in range(5):
            try:
                t0 = time.time()
                p = hf_hub_download(repo_id=REPO, filename=fn, repo_type="dataset",
                                    local_dir=DEST)
                print(f"OK {fn} -> {p} ({time.time()-t0:.0f}s, {os.path.getsize(p)/1e6:.0f} MB)")
                got.append(p)
                break
            except Exception as e:
                print(f"  attempt {attempt} for {fn} failed: {repr(e)[:160]}")
                time.sleep(2 ** attempt)
        else:
            print(f"GAVE UP on {fn}")
    print(f"\nstaged {len(got)} shards under {DEST}")
    # quick read sanity (local parquet, no network)
    if got:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=got, split="train", streaming=True)
        ex = next(iter(ds))
        print("sample row keys:", list(ex.keys()), "| text[:80]:", repr(ex.get("text", "")[:80]))


if __name__ == "__main__":
    main()
