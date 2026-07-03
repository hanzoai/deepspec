"""Convert a Lux engine target-hidden-state dump into a DeepSpec target cache.

The Rust engine (separate workstream) dumps, per training sample:

  * ``input_ids``                 [seq]            int64/int32
  * ``target_hidden_states``      [seq, L, H]      bf16   (L captured layers)
  * ``target_last_hidden_states`` [seq, H]         bf16
  * (optional) ``loss_mask``      [seq]            any int/bool
  * (optional) ``attention_mask`` [seq]            any int/bool

alongside an ``index.jsonl`` (one JSON object per line) that points at each
per-sample ``*.safetensors`` shard via a ``file`` / ``path`` / ``safetensors``
key.

This script re-emits those samples in DeepSpec's canonical *target cache*
protocol (version 2) by driving DeepSpec's OWN writer/finalizer classes -- it
does not reimplement the binary layout. The output directory is directly
consumable by ``deepspec.data.CacheDataset`` and ``scripts/train/train.sh``.

``target_hidden_states`` is stored as ``[seq, L * H]`` with the L captured
layers concatenated along the hidden axis in ``--target-layer-ids`` order --
byte-identical to what ``run_target_forward_with_hooks`` in
``scripts/data/prepare_target_cache.py`` produces
(``torch.cat([captured[id] for id in target_layer_ids], dim=-1)``).

Single process, no torch.distributed: this is the ``world_size == 1``
specialization of ``scripts/data/prepare_target_cache.py``'s finalize flow.
"""

import argparse
import json
import os

import torch
from safetensors.torch import load_file

from deepspec.data.target_cache_dataset import (
    AsyncTargetCacheWriter,
    LocalCacheWriteSummary,
    atomic_json_dump,
    build_global_target_cache_shard_map,
    build_target_cache_manifest,
    cleanup_target_cache_tmp_dir,
    finalize_target_cache_index,
    load_local_cache_write_summary,
    load_target_cache_manifest,
    prepare_target_cache_output_dir,
    rename_local_target_cache_shards,
    write_target_cache_manifest,
)


_FILE_KEYS = ("file", "path", "safetensors", "sample", "shard")
_INPUT_IDS_KEYS = ("input_ids", "input_id", "ids", "tokens")
_HIDDEN_KEYS = ("target_hidden_states", "hidden_states", "target_hidden")
_LAST_HIDDEN_KEYS = (
    "target_last_hidden_states",
    "last_hidden_states",
    "target_last_hidden",
)
_LOSS_MASK_KEYS = ("loss_mask", "loss_masks")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dump-dir",
        required=True,
        help="Engine-dump directory containing index.jsonl + *.safetensors.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination target-cache directory (must be new/empty).",
    )
    parser.add_argument(
        "--index-name",
        default="index.jsonl",
        help="Index file name inside --dump-dir (default: index.jsonl).",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=4096,
        help="Target hidden dimension H (V4-Flash: 4096).",
    )
    parser.add_argument(
        "--target-layer-ids",
        default="1,11,21,31,41",
        help="Comma-separated captured target layer ids, ascending "
        "(V4-Flash: 1,11,21,31,41).",
    )
    parser.add_argument(
        "--target-model-name-or-path",
        required=True,
        help="Recorded in the manifest; MUST equal "
        "config.model.target_model_name_or_path used at train time "
        "(validate_train_cache asserts string equality).",
    )
    parser.add_argument(
        "--loss-mask-mode",
        choices=("ones_except_first", "all_ones", "from_dump"),
        default="ones_except_first",
        help="How to build loss_mask. 'ones_except_first' (default): 1 "
        "everywhere except position 0 -- position 0 has no left context so it "
        "is never a valid anchor nor a first draft target "
        "(see build_anchor_candidate_mask). 'from_dump': use the dump's "
        "loss_mask (error if a sample lacks one).",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=64 * 1024**3,
        help="Max bytes per output shard before rolling to a new one.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only convert the first N index entries (debug).",
    )
    return parser.parse_args()


def _first_present(mapping, keys, what, where):
    for key in keys:
        if key in mapping:
            return mapping[key]
    raise KeyError(f"{what} not found in {where}; looked for keys {keys}.")


def _resolve_sample_path(dump_dir, record, line_no):
    file_ref = None
    for key in _FILE_KEYS:
        if key in record:
            file_ref = record[key]
            break
    if file_ref is None:
        raise KeyError(
            f"index line {line_no}: no shard reference; expected one of "
            f"{_FILE_KEYS} in {record!r}."
        )
    path = str(file_ref)
    if not os.path.isabs(path):
        path = os.path.join(dump_dir, path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"index line {line_no}: missing sample file {path}.")
    return path


def _build_loss_mask(mode, seq_len, tensors, device):
    if mode == "from_dump":
        loss_mask = _first_present(
            tensors, _LOSS_MASK_KEYS, "loss_mask", "sample tensors"
        )
        return loss_mask.reshape(-1).to(dtype=torch.uint8)
    loss_mask = torch.ones(seq_len, dtype=torch.uint8, device=device)
    if mode == "ones_except_first" and seq_len > 0:
        loss_mask[0] = 0
    return loss_mask


def _load_sample(path, *, hidden_size, num_target_layers, loss_mask_mode):
    tensors = load_file(path)
    input_ids = _first_present(tensors, _INPUT_IDS_KEYS, "input_ids", path).reshape(-1)
    seq_len = int(input_ids.shape[0])
    assert seq_len > 0, f"{path}: empty input_ids."

    hidden = _first_present(tensors, _HIDDEN_KEYS, "target_hidden_states", path)
    expected_hidden_numel = seq_len * num_target_layers * hidden_size
    assert hidden.numel() == expected_hidden_numel, (
        f"{path}: target_hidden_states has {hidden.numel()} elements, expected "
        f"{expected_hidden_numel} = seq({seq_len}) * layers({num_target_layers}) "
        f"* hidden({hidden_size}). Got shape {tuple(hidden.shape)}."
    )
    # [seq, L, H] or already [seq, L*H] -> [seq, L*H] (row-major keeps
    # per-layer blocks contiguous, matching target_layer_ids concat order).
    target_hidden_states = hidden.reshape(seq_len, num_target_layers * hidden_size)

    last_hidden = _first_present(
        tensors, _LAST_HIDDEN_KEYS, "target_last_hidden_states", path
    )
    assert last_hidden.numel() == seq_len * hidden_size, (
        f"{path}: target_last_hidden_states has {last_hidden.numel()} elements, "
        f"expected {seq_len * hidden_size} = seq({seq_len}) * hidden({hidden_size})."
    )
    target_last_hidden_states = last_hidden.reshape(seq_len, hidden_size)

    loss_mask = _build_loss_mask(loss_mask_mode, seq_len, tensors, input_ids.device)
    assert loss_mask.shape[0] == seq_len, (
        f"{path}: loss_mask length {loss_mask.shape[0]} != seq_len {seq_len}."
    )
    attention_mask = torch.ones(seq_len, dtype=torch.uint8, device=input_ids.device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "target_hidden_states": target_hidden_states,
        "target_last_hidden_states": target_last_hidden_states,
    }


def _read_index(index_path, limit):
    records = []
    with open(index_path, "r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle):
            raw = raw.strip()
            if not raw:
                continue
            records.append((line_no, json.loads(raw)))
            if limit is not None and len(records) >= limit:
                break
    assert records, f"No records read from {index_path}."
    return records


def main():
    args = parse_args()
    dump_dir = os.path.abspath(args.dump_dir)
    output_dir = os.path.abspath(args.output_dir)
    index_path = os.path.join(dump_dir, args.index_name)
    assert os.path.exists(index_path), f"Missing index file: {index_path}"

    target_layer_ids = [int(x) for x in args.target_layer_ids.split(",") if x != ""]
    assert target_layer_ids == sorted(target_layer_ids), (
        f"--target-layer-ids must be ascending, got {target_layer_ids}."
    )
    num_target_layers = len(target_layer_ids)
    hidden_size = int(args.hidden_size)

    records = _read_index(index_path, args.limit)
    print(
        f"Converting {len(records)} samples from {dump_dir}\n"
        f"  hidden_size={hidden_size} target_layer_ids={target_layer_ids} "
        f"loss_mask_mode={args.loss_mask_mode}\n"
        f"  -> {output_dir}",
        flush=True,
    )

    prepare_target_cache_output_dir(output_dir)
    rank_dir = os.path.join(output_dir, "_tmp", "rank_0")
    os.makedirs(rank_dir, exist_ok=True)

    writer = AsyncTargetCacheWriter(
        rank_dir=rank_dir,
        max_shard_bytes=int(args.max_shard_bytes),
    )
    try:
        for idx, (line_no, record) in enumerate(records):
            path = _resolve_sample_path(dump_dir, record, line_no)
            sample = _load_sample(
                path,
                hidden_size=hidden_size,
                num_target_layers=num_target_layers,
                loss_mask_mode=args.loss_mask_mode,
            )
            writer.write_sample(**sample)
            if (idx + 1) % 100 == 0 or (idx + 1) == len(records):
                print(f"  wrote {idx + 1}/{len(records)} samples", flush=True)
    finally:
        writer.close()

    summary = LocalCacheWriteSummary(
        global_rank=0,
        source_sample_start=0,
        source_sample_end=len(records),
        num_local_samples=writer.num_local_samples,
        num_local_shards=len(writer.local_shard_files),
        local_shard_files=list(writer.local_shard_files),
    )
    atomic_json_dump(summary.to_json(), os.path.join(rank_dir, "summary.json"))

    summaries = [load_local_cache_write_summary(rank_dir)]
    shard_map, shards = build_global_target_cache_shard_map(summaries)
    rename_local_target_cache_shards(
        output_dir=output_dir,
        rank_dir=rank_dir,
        summary=summaries[0],
        shard_map=shard_map,
    )
    num_valid_samples = finalize_target_cache_index(
        output_dir=output_dir,
        summaries=summaries,
        shard_map=shard_map,
    )
    manifest = build_target_cache_manifest(
        num_samples=num_valid_samples,
        shards=shards,
        target_layer_ids=target_layer_ids,
        hidden_size=hidden_size,
        extra_fields={
            "target_model_name_or_path": str(args.target_model_name_or_path),
            "source_engine_dump": dump_dir,
            "loss_mask_mode": args.loss_mask_mode,
            "converter": "scripts/data/convert_engine_dump.py",
        },
    )
    write_target_cache_manifest(output_dir=output_dir, manifest=manifest)
    cleanup_target_cache_tmp_dir(output_dir)

    # Re-load through DeepSpec's own validator to prove the cache is valid.
    load_target_cache_manifest(output_dir)
    print(
        f"Wrote valid target cache: {num_valid_samples} samples, "
        f"{len(shards)} shard(s) at {output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
