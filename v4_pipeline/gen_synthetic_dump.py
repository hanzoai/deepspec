"""Generate a synthetic Lux engine target-hidden-state dump for smoke testing.

Mimics exactly what the Rust engine tool (separate workstream) emits: an
``index.jsonl`` plus per-sample ``sample_*.safetensors`` files with

  * ``input_ids``                 [seq]        int64 (values < vocab_size)
  * ``target_hidden_states``      [seq, L, H]  bf16
  * ``target_last_hidden_states`` [seq, H]     bf16

Random data only -- this exercises the CONVERTER + TRAINING pipeline shapes,
not model quality.
"""

import argparse
import json
import os

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=4096)
    parser.add_argument("--num-target-layers", type=int, default=5)
    parser.add_argument("--vocab-size", type=int, default=129280)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    torch.manual_seed(args.seed)

    index_path = os.path.join(args.output_dir, "index.jsonl")
    with open(index_path, "w", encoding="utf-8") as index_handle:
        for i in range(args.num_samples):
            file_name = f"sample_{i:05d}.safetensors"
            tensors = {
                "input_ids": torch.randint(
                    0, args.vocab_size, (args.seq_len,), dtype=torch.int64
                ),
                "target_hidden_states": torch.randn(
                    args.seq_len, args.num_target_layers, args.hidden_size
                ).to(torch.bfloat16),
                "target_last_hidden_states": torch.randn(
                    args.seq_len, args.hidden_size
                ).to(torch.bfloat16),
            }
            save_file(tensors, os.path.join(args.output_dir, file_name))
            index_handle.write(
                json.dumps({"file": file_name, "seq_len": args.seq_len}) + "\n"
            )

    print(
        f"Wrote {args.num_samples} synthetic samples "
        f"(seq={args.seq_len}, layers={args.num_target_layers}, "
        f"hidden={args.hidden_size}, vocab={args.vocab_size}) to {args.output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
