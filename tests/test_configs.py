"""Every shipped YAML must load, validate, and mean what the pipeline assumes it means."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from cpt_m2d2.config import (
    CPTTrainConfig,
    DataConfig,
    ModelConfig,
    SFTTrainConfig,
    load_config,
)

ROOT = Path(os.environ.get("CPT_ROOT", Path(__file__).resolve().parents[1]))
CONFIGS = ROOT / "configs"


def test_model_config_loads():
    cfg = load_config(CONFIGS / "model/qwen3_4b.yaml", ModelConfig)
    assert cfg.torch_dtype == "bfloat16"
    assert cfg.trust_remote_code is False
    assert cfg.use_cache is False
    assert cfg.resolved_tokenizer_path == cfg.name_or_path
    assert "${" not in cfg.name_or_path, "env expansion did not run"


def test_base_model_vocab_forces_uint32():
    cfg = load_config(CONFIGS / "model/qwen3_4b.yaml", ModelConfig)
    config_json = Path(cfg.name_or_path) / "config.json"
    if not config_json.is_file():
        pytest.skip("base model config.json not present")
    vocab = json.loads(config_json.read_text())["vocab_size"]
    assert vocab > 65535, "uint32 packing is only justified above 65535"
    assert vocab >= cfg.expected_min_vocab_size


def test_data_config_loads_and_is_consistent():
    cfg = load_config(CONFIGS / "data/m2d2_stages.yaml", DataConfig)
    assert cfg.seq_len == 4096
    assert cfg.eval_max_tokens == 2_000_000
    assert cfg.replay.enabled and cfg.replay.source
    assert cfg.replay.source not in cfg.training_domains(), (
        "the replay source is also a training stage, so it is not an out-of-branch control"
    )
    assert len(set(cfg.training_domains())) == len(cfg.training_domains())


def test_configured_domains_exist_on_disk():
    cfg = load_config(CONFIGS / "data/m2d2_stages.yaml", DataConfig)
    raw = Path(cfg.raw_dir)
    if not raw.is_dir():
        pytest.skip(f"raw dir {raw} not present")
    for domain in cfg.domains_to_pack():
        assert (raw / domain / "train.txt").is_file(), f"{domain} has no train.txt under {raw}"


def test_misspelled_upstream_domain_is_preserved():
    """`Human_activites` is misspelled upstream and is correct as written."""
    text = (CONFIGS / "data/m2d2_stages.yaml").read_text()
    assert "Human_activites__Human_activities" in text
    assert "Human_activities__Human_activities" not in text


def test_train_config_defaults():
    cfg = load_config(CONFIGS / "train/cpt_base.yaml", CPTTrainConfig)
    assert cfg.num_train_epochs == 1.0
    assert cfg.save_strategy == "no"
    assert cfg.save_only_model is True
    assert cfg.keep_stage_checkpoints is False
    assert cfg.keep_stage_checkpoints_min_free_gb == 40.0
    assert cfg.max_stage_hours == 3.0
    assert cfg.gradient_checkpointing is True
    assert cfg.bf16 is True


def test_tokens_per_optimizer_step():
    cfg = load_config(CONFIGS / "train/cpt_base.yaml", CPTTrainConfig)
    assert cfg.tokens_per_optimizer_step(seq_len=4096, world_size=2) == 524_288


def test_multi_epoch_is_refused_by_default():
    with pytest.raises(ValueError, match="inflate measured forgetting"):
        CPTTrainConfig(num_train_epochs=3)
    # ...but is reachable with an explicit acknowledgement.
    CPTTrainConfig(num_train_epochs=3, i_understand_multi_epoch_inflates_forgetting=True)


def test_unknown_key_is_an_error_not_a_silent_no_op():
    with pytest.raises(ValueError):
        CPTTrainConfig(learning_rate=1e-5, lr_schedule_type="cosine")  # typo in the key


def test_sft_config_loads():
    cfg = load_config(CONFIGS / "train/sft_base.yaml", SFTTrainConfig)
    assert cfg.completion_only_loss is True


@pytest.mark.parametrize(
    "name", ["ds_zero2.yaml", "ds_zero3.yaml", "fsdp2_grad_op.yaml", "fsdp2_full.yaml"]
)
def test_accelerate_configs_parse(name: str):
    cfg = yaml.safe_load((CONFIGS / "accel" / name).read_text())
    assert cfg["mixed_precision"] == "bf16"
    assert cfg["num_processes"] == 2
    assert cfg["use_cpu"] is False


def test_zero3_gathers_weights_on_save():
    """Without this, save_only_model writes shards from_pretrained cannot read - and with no
    intermediate checkpoints that loss is unrecoverable."""
    ds = json.loads((CONFIGS / "accel/ds_zero3.json").read_text())
    zero = ds["zero_optimization"]
    assert zero["stage"] == 3
    assert zero["stage3_gather_16bit_weights_on_model_save"] is True
    assert zero["stage3_param_persistence_threshold"] == 1e5
    assert zero["overlap_comm"] is True
    assert zero["contiguous_gradients"] is True


def test_zero2_is_the_default_and_has_no_offload():
    ds = json.loads((CONFIGS / "accel/ds_zero2.json").read_text())
    assert ds["zero_optimization"]["stage"] == 2
    assert ds["zero_optimization"]["offload_optimizer"]["device"] == "none"
    assert ds["zero_optimization"]["offload_param"]["device"] == "none"


@pytest.mark.parametrize(
    ("name", "reshard"), [("fsdp2_grad_op.yaml", False), ("fsdp2_full.yaml", True)]
)
def test_fsdp2_sharding_and_autowrap(name: str, reshard: bool):
    cfg = yaml.safe_load((CONFIGS / "accel" / name).read_text())["fsdp_config"]
    model_cfg = load_config(CONFIGS / "model/qwen3_4b.yaml", ModelConfig)
    assert cfg["fsdp_version"] == 2
    assert cfg["fsdp_reshard_after_forward"] is reshard
    assert cfg["fsdp_auto_wrap_policy"] == "TRANSFORMER_BASED_WRAP"
    assert cfg["fsdp_transformer_layer_cls_to_wrap"] == model_cfg.decoder_layer_cls
    assert cfg["fsdp_offload_params"] is False
    assert cfg["fsdp_state_dict_type"] == "FULL_STATE_DICT"


def test_decoder_layer_class_exists_in_transformers():
    from transformers.models.qwen3 import modeling_qwen3

    model_cfg = load_config(CONFIGS / "model/qwen3_4b.yaml", ModelConfig)
    assert hasattr(modeling_qwen3, model_cfg.decoder_layer_cls)