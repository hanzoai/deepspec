"""DSpark draft config for a DeepSeek-V4-Flash (quantized) serving target.

Adapted from ``config/dspark/dspark_qwen3_4b.py``. The DRAFT is a standard
DSpark dense drafter (5 layers, block 7, Markov + confidence heads); only the
target it distills from changes. The target model here is DeepSeek-V4-Flash
served QUANTIZED, from which the Lux engine dumps per-token hidden states that
``scripts/data/convert_engine_dump.py`` packs into a DeepSpec target cache.

V4-Flash target dimensions (derived from the target's HF config at train time,
listed here for reference):
    hidden_size           = 4096
    vocab_size            = 129280
    num_hidden_layers     = 43      (target layers; captured ids below)

INTEGRATION TODO (target tokenizer / embeddings):
    ``deepspec.trainer.base_trainer.BaseTrainer.build_models`` loads
    ``target_model_name_or_path`` three ways:
      1. ``AutoTokenizer.from_pretrained`` (tokenizer for logging/eval only),
      2. ``AutoConfig.from_pretrained``    (to derive the draft dims via
         ``build_qwen3_draft_config`` -- deepcopies the target config),
      3. ``AutoModelForCausalLM.from_pretrained`` (to copy + FREEZE the draft's
         ``embed_tokens`` and ``lm_head`` from the target).
    We currently only have the V4 tokenizer inside a GGUF. A real training run
    therefore needs a LOCAL HF-format directory at ``target_model_name_or_path``
    containing at least ``config.json`` + tokenizer files + the embedding and
    lm_head weights (full weights not required if a trimmed checkpoint exposing
    just the input/output embeddings is provided). Because the target is
    DeepSeek (not Qwen3), that ``config.json`` must expose the Qwen3-style
    fields the dense drafter reads (``num_attention_heads``,
    ``num_key_value_heads``, ``head_dim``, ``intermediate_size``,
    ``rms_norm_eps``, ``rope_parameters``, ``max_position_embeddings``), or a
    tiny target-config adapter must synthesize a Qwen3Config with V4 dims. The
    toy smoke (``v4_pipeline/smoke_train.py``) bypasses all three loads by
    constructing a synthetic Qwen3 target config with the V4 dims above.

The manifest's ``target_model_name_or_path`` (written by the converter) MUST
equal ``model.target_model_name_or_path`` below -- ``validate_train_cache``
asserts string equality.
"""

import os
from deepspec.trainer import Qwen3DSparkTrainer

BASE_TB_DIR = os.path.expanduser("~/tensorboard")
BASE_CKPT_DIR = os.path.expanduser("~/checkpoints")
project_name = "deepspec"
exp_name = "dspark_block7_v4flash_quant"
seed = 42

# Local HF-format directory for the DeepSeek-V4-Flash target (see
# INTEGRATION TODO above). Must match the converter's
# --target-model-name-or-path.
TARGET_MODEL_PATH = os.path.expanduser("~/models/deepseek-v4-flash")

model = dict(
    target_model_name_or_path=TARGET_MODEL_PATH,
    block_size=7,
    num_draft_layers=5,
    # 5 captured target layers out of 43 (must be strictly ascending and each
    # in [-1, 42]; -1 would denote the embedding output).
    target_layer_ids=[1, 11, 21, 31, 41],
    # Mask/noise token fed at draft positions (create_noise_embed ->
    # embed_tokens lookup ONLY). Requirements: 0 <= id < vocab_size (129280)
    # and the tokenizer must never EMIT it, so a masked draft position is never
    # confused with a real token. 129279 = vocab_size - 1 is the top-most id and
    # sits in DeepSeek's reserved/special band -> a safe, unemitted placeholder.
    # TODO: once the real V4 tokenizer is available, confirm this id is reserved
    # (ideally point it at a dedicated <|mask|>/pad reserved token).
    mask_token_id=129279,
    num_anchors=512,

    ## markov head
    markov_rank=256,
    markov_head_type='vanilla',

    ## confidence head
    confidence_head_alpha=1.0,
    confidence_head_with_markov=True,

    ## loss
    loss_decay_gamma=4.0,
    ce_loss_alpha=0.1,
    l1_loss_alpha=0.9,
)

train = dict(
    trainer_cls=Qwen3DSparkTrainer,
    lr=6.0e-4,
    warmup_ratio=0.04,
    weight_decay=0.0,
    precision="bf16",
    local_batch_size=1,
    global_batch_size=512,
    num_train_epochs=10,
    max_train_steps=None,
    max_grad_norm=1.0,
    sharding_strategy="no_shard",
    torch_compile=True,
)

logging = dict(
    logging_steps=10,
    checkpointing_steps=3000,
)

data = dict(
    target_cache_path=None,
    chat_template="qwen",
    max_length=4096,
    num_workers=4,
)


def finalize_cfg(cfg):
    logging_cfg = dict(cfg["logging"])
    project_name = str(cfg['project_name'])
    exp_name = str(cfg["exp_name"])
    logging_cfg["checkpoint_dir"] = os.path.join(BASE_CKPT_DIR, project_name, exp_name)
    logging_cfg["tensorboard_dir"] = os.path.join(BASE_TB_DIR, project_name, exp_name)
    cfg["logging"] = logging_cfg

    return cfg
