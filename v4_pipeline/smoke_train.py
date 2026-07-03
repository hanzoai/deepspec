"""Toy end-to-end DSpark training smoke for the DeepSeek-V4-Flash config.

Proves the PYTHON training pipeline runs on this box:
    CacheDataset -> CacheCollator -> Qwen3DSparkModel forward
    -> compute_dspark_loss -> backward -> BF16Optimizer.step

It reuses the REAL DeepSpec components (config loader, draft-config builder,
model, dataset, collator, loss, optimizer). It only bypasses
``BaseTrainer.build_models``' three ``from_pretrained`` loads of the V4 target
(tokenizer / AutoConfig / AutoModelForCausalLM), which need a local HF-format
V4 checkpoint we do not have yet (see dspark_v4flash_quant.py INTEGRATION TODO).
Bypass = (a) synthesize a Qwen3 target config carrying the real V4 dims, and
(b) randomly init + FREEZE embed_tokens/lm_head instead of copying them from the
target (frozen == exactly what build_models does after copying).

SMOKE REDUCTIONS vs the shipped config (memory only; every other knob is real):
    --num-draft-layers   (default 2, config ships 5)
    --num-anchors        (default 16, config ships 512)
    --intermediate-size  (default 4096; the true V4 dense MLP width is larger)
Real & unchanged: hidden 4096, vocab 129280, block_size 7, markov rank 256
(vanilla), confidence head on, target_layer_ids [1,11,21,31,41], bf16,
flex_attention, BF16Optimizer, compute_dspark_loss (ce+tv+confidence).
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from transformers import Qwen3Config

from deepspec.data import CacheCollator, CacheDataset, validate_train_cache
from deepspec.modeling.dspark.loss import compute_dspark_loss
from deepspec.modeling.dspark.qwen3 import Qwen3DSparkModel
from deepspec.modeling.dspark.qwen3.config import build_draft_config
from deepspec.utils import BF16Optimizer, load_config, seed_all


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-draft-layers", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=16)
    parser.add_argument("--intermediate-size", type=int, default=4096)
    return parser.parse_args()


def _init_single_process_group():
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29577")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    if not dist.is_initialized():
        # gloo: the process group is only used by compute_dspark_loss for a
        # world_size>1 all-reduce (skipped at world_size==1); CUDA tensors are
        # never sent over it.
        dist.init_process_group(backend="gloo", rank=0, world_size=1)


def _build_target_config(cfg, args):
    """Synthetic Qwen3 target config carrying the real V4-Flash dims.

    Stands in for AutoConfig.from_pretrained(target) which we cannot run
    without a local HF V4 checkpoint. num_hidden_layers (43) must exceed
    max(target_layer_ids) so validate_target_layer_ids passes.
    """
    return Qwen3Config(
        hidden_size=4096,
        vocab_size=129280,
        num_hidden_layers=43,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=int(args.intermediate_size),
        max_position_embeddings=4096,
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)
    seed_all(int(cfg.seed))
    _init_single_process_group()

    device = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    precision_dtype = torch.bfloat16

    model_args = cfg.model
    # Apply the documented smoke-only memory reductions.
    model_args.num_draft_layers = int(args.num_draft_layers)
    model_args.num_anchors = int(args.num_anchors)

    target_config = _build_target_config(cfg, args)
    draft_config = build_draft_config(target_config=target_config, model_args=model_args)
    model = Qwen3DSparkModel(draft_config).to(device=device, dtype=precision_dtype)
    # build_models copies embed_tokens/lm_head from the target then freezes them.
    # We lack the target checkpoint, so we keep the random init and just freeze
    # (identical trainability; pipeline-only smoke).
    model.set_embedding_head_trainable(False)
    model.train()

    dataset = CacheDataset(cache_dir=args.cache_dir)
    validate_train_cache(
        train_dataset=dataset,
        draft_model=model,
        target_model_name_or_path=model_args.target_model_name_or_path,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(cfg.train.local_batch_size),
        shuffle=True,
        collate_fn=CacheCollator(),
        drop_last=True,
    )

    optimizer = BF16Optimizer(
        model,
        lr=float(cfg.train.lr),
        total_steps=int(args.steps),
        warmup_ratio=float(cfg.train.warmup_ratio),
        weight_decay=float(cfg.train.weight_decay),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(
        f"[smoke] device={device} draft_layers={model_args.num_draft_layers} "
        f"num_anchors={model_args.num_anchors} intermediate={args.intermediate_size}\n"
        f"[smoke] params: trainable={trainable/1e6:.1f}M frozen(embed+head)="
        f"{frozen/1e6:.1f}M | dataset={len(dataset)} samples",
        flush=True,
    )

    data_iter = iter(dataloader)
    t0 = time.time()
    for step in range(int(args.steps)):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)
        batch = {
            key: value.to(device, non_blocking=True) for key, value in batch.items()
        }
        # Same boundary cast as the real path (cuda_prefetcher.move_batch_to_device):
        # embedding lookup requires int64, cache stores int32.
        if batch["input_ids"].dtype != torch.long:
            batch["input_ids"] = batch["input_ids"].to(torch.long)
        outputs = model(
            input_ids=batch["input_ids"],
            target_hidden_states=batch["target_hidden_states"],
            loss_mask=batch["loss_mask"],
            target_last_hidden_states=batch["target_last_hidden_states"],
        )
        loss = compute_dspark_loss(
            outputs=outputs,
            loss_decay_gamma=cfg.model.loss_decay_gamma,
            ce_loss_alpha=float(cfg.model.ce_loss_alpha),
            l1_loss_alpha=float(cfg.model.l1_loss_alpha),
            confidence_head_alpha=float(cfg.model.confidence_head_alpha),
        )
        loss.backward()
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        peak = (
            torch.cuda.max_memory_allocated() / 1024**3
            if device.type == "cuda"
            else 0.0
        )
        print(
            f"[smoke] step {step + 1}/{args.steps} loss={loss.item():.5f} "
            f"lr={optimizer.get_learning_rate():.3e} "
            f"peak_vram={peak:.2f}GB dt={time.time() - t0:.1f}s",
            flush=True,
        )

    dataset.close()
    if dist.is_initialized():
        dist.destroy_process_group()
    print("[smoke] OK: dataloader -> forward -> loss -> backward -> step verified.")


if __name__ == "__main__":
    main()
