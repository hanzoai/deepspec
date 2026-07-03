#!/usr/bin/env python3
"""open-perfectblend parquet -> JSONL for hanzo-engine v4_cache_dump.

Each line: {"text": human + "\n\n" + gpt, "prompt_frac": len(human)/len(text), "source": ...}
prompt_frac lets the converter zero loss_mask over the prompt region by ratio.
"""
import argparse, glob, json
import pyarrow.parquet as pq

ap = argparse.ArgumentParser()
ap.add_argument("--dataset-dir", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--min-chars", type=int, default=200)
ap.add_argument("--max-chars", type=int, default=8000)
args = ap.parse_args()

fs = sorted(glob.glob(f"{args.dataset_dir}/**/*.parquet", recursive=True))
out = open(args.out, "w")
n = kept = 0
for fp in fs:
    for batch in pq.ParquetFile(fp).iter_batches(batch_size=2000):
        for row in batch.to_pylist():
            n += 1
            conv = row["conversations"]
            h = next((t["value"] for t in conv if t["from"] == "human"), None)
            g = next((t["value"] for t in conv if t["from"] == "gpt"), None)
            if not h or not g:
                continue
            text = h.strip() + "\n\n" + g.strip()
            if not (args.min_chars <= len(text) <= args.max_chars):
                continue
            kept += 1
            out.write(json.dumps({"text": text, "prompt_frac": round(len(h) / len(text), 4),
                                  "source": row.get("source", "")}) + "\n")
    print(f"{fp.split('/')[-1]}: total={n} kept={kept}", flush=True)
out.close()
print(f"DONE: {kept}/{n} -> {args.out}")
