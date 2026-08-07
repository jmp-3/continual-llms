"""Pydantic schemas for every YAML config in ``configs/``.

Every schema is ``extra="forbid"``: a typo in a YAML key is an error at load time rather
than a silently-ignored setting that costs a multi-hour run.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .env import expand, load_env

__all__ = [
    "ModelConfig",
    "StageSpec",
    "ReplayConfig",
    "DataConfig",
    "CPTTrainConfig",
    "SFTTrainConfig",
    "load_config",
    "expand_env_in_obj",
]


class _Base(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=False)


# --------------------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------------------


class ModelConfig(_Base):
    """``configs/model/qwen3_4b.yaml``."""

    name_or_path: str
    tokenizer_path: str | None = None
    attn_implementation: str = "sdpa"
    torch_dtype: Literal["bfloat16"] = "bfloat16"
    trust_remote_code: bool = False
    use_cache: bool = False
    # Qwen3 ships vocab_size=151936. Anything above 65535 forces uint32 packed arrays;
    # this is asserted rather than assumed, in both tokenize_pack and the tests.
    expected_min_vocab_size: int = 65536
    # Used by the accelerate configs' transformer auto-wrap policy.
    decoder_layer_cls: str = "Qwen3DecoderLayer"

    @property
    def resolved_tokenizer_path(self) -> str:
        return self.tokenizer_path or self.name_or_path


# --------------------------------------------------------------------------------------
# data / curriculum
# --------------------------------------------------------------------------------------


class StageSpec(_Base):
    name: str
    domain: str
    #: Cap on *main-domain* training tokens for this stage. ``None`` means the whole split.
    #: Replay tokens are added on top of this cap (see :class:`ReplayConfig`).
    max_tokens: int | None = None


class ReplayConfig(_Base):
    enabled: bool = True
    #: Domain to draw replay blocks from. Kept out of ``stages`` so it stays a clean
    #: out-of-branch control.
    source: str | None = None
    #: Target fraction of each stage's *total* training tokens that come from ``source``.
    ratio: float = Field(default=0.05, ge=0.0, lt=1.0)
    #: Blocks are drawn without replacement within a stage. If the replay pool is smaller
    #: than the target, the realized fraction drops; set this to fail instead of warn.
    strict_ratio: bool = False
    #: How far the realized fraction may drift from ``ratio`` before it is a hard error
    #: (only consulted when ``strict_ratio`` is true).
    ratio_tolerance: float = Field(default=0.005, ge=0.0)

    @model_validator(mode="after")
    def _source_required_when_enabled(self) -> ReplayConfig:
        if self.enabled and not self.source:
            raise ValueError("replay.enabled is true but replay.source is unset")
        return self


class DataConfig(_Base):
    """``configs/data/m2d2_stages.yaml``."""

    seq_len: int = Field(default=4096, gt=0)
    raw_dir: str = "${CL_RAW_DIR}"
    packed_dir: str = "${CL_PACKED_DIR}"
    train_split: str = "train"
    eval_split: str = "valid"
    stages: list[StageSpec]
    replay: ReplayConfig = ReplayConfig(enabled=False, source=None)
    #: ``["ALL"]`` expands to every domain found under ``packed_dir``.
    eval_domains: list[str] = ["ALL"]
    #: Hard cap per eval set. The matrix costs O(stages x domains) evaluations, so full
    #: eval splits are unaffordable; 2M tokens keeps a row to ~half an hour on 2 GPUs.
    eval_max_tokens: int = Field(default=2_000_000, gt=0)
    #: How a stage's ``max_tokens`` cap selects blocks from the packed domain.
    #: ``"head"`` takes the first N blocks - deterministic, sequential reads, and identical
    #: across curriculum permutations. ``"random"`` samples N blocks without replacement,
    #: which is more representative if a domain's file has any ordering structure.
    max_tokens_sampling: Literal["head", "random"] = "head"
    #: Number of consecutive blank lines that separates two documents. ``"auto"`` picks
    #: per-domain from a blank-run histogram of the file head (see tokenize_pack).
    doc_delimiter_blank_lines: Literal["auto"] | int = "auto"
    #: Per-domain override of the above, keyed by domain name.
    doc_delimiter_overrides: dict[str, int] = {}
    #: Extra domains to pack for evaluation only (never trained on).
    eval_only_domains: list[str] = []

    @field_validator("stages")
    @classmethod
    def _stage_names_unique(cls, stages: list[StageSpec]) -> list[StageSpec]:
        names = [s.name for s in stages]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate stage names: {names}")
        if not stages:
            raise ValueError("at least one stage is required")
        return stages

    def training_domains(self) -> list[str]:
        return [s.domain for s in self.stages]

    def delimiter_for(self, domain: str) -> Literal["auto"] | int:
        return self.doc_delimiter_overrides.get(domain, self.doc_delimiter_blank_lines)

    def domains_to_pack(self) -> list[str]:
        """Every domain that needs a packed artifact, in a stable order."""
        seen: list[str] = []
        for name in [
            *self.training_domains(),
            *( [self.replay.source] if self.replay.enabled and self.replay.source else [] ),
            *self.eval_only_domains,
            *[d for d in self.eval_domains if d != "ALL"],
        ]:
            if name and name not in seen:
                seen.append(name)
        return seen


# --------------------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------------------


class CPTTrainConfig(_Base):
    """``configs/train/cpt_base.yaml``."""

    # --- passed through to transformers.TrainingArguments -----------------------------
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 64
    gradient_checkpointing: bool = True
    bf16: bool = True
    optim: str = "adamw_torch"
    learning_rate: float = 2.0e-5
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.01
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: float = 1.0
    save_strategy: Literal["no"] = "no"
    save_only_model: bool = True
    logging_steps: int = 5
    seed: int = 1234
    dataloader_num_workers: int = 2

    # --- pipeline policy ---------------------------------------------------------------
    runs_dir: str = "${CL_RUNS_DIR}"
    run_id: str | None = None
    #: Copy each stage's weights to ``runs/<run_id>/stage_<n>/`` after the atomic swap.
    #: ~8 GB per stage for a 4B bf16 model.
    keep_stage_checkpoints: bool = False
    #: Refuse the copy above if it would drop free space below this many GB.
    keep_stage_checkpoints_min_free_gb: float = 40.0
    #: Wall-clock ceiling per stage. There are no mid-stage checkpoints, so a crash loses
    #: the whole stage; ``--allow-long-stage`` is the only way past this.
    max_stage_hours: float = 3.0
    #: A-priori throughput used to estimate stage wall time *before* stage 1 has produced
    #: a measurement. Replaced by the measured value from ``throughput.json`` thereafter.
    assumed_tokens_per_second: float = 2200.0
    #: Optimizer steps over which to measure real throughput once training starts.
    throughput_measure_steps: int = 3
    #: Optimizer steps to skip before measuring (compile/allocator warmup).
    throughput_warmup_steps: int = 2
    #: Evaluate the untouched base model as row 0 of the matrix. Required for FWT.
    eval_baseline: bool = True
    #: Guard against the multi-epoch failure mode described below.
    i_understand_multi_epoch_inflates_forgetting: bool = False

    @model_validator(mode="after")
    def _one_epoch_default(self) -> CPTTrainConfig:
        if self.num_train_epochs > 1.0 and not self.i_understand_multi_epoch_inflates_forgetting:
            raise ValueError(
                "num_train_epochs > 1 is refused by default. Domain corpora are small "
                "relative to a 4B model; multiple epochs at 2e-5 memorize the domain and "
                "inflate measured forgetting, which corrupts the quantity under study. "
                "If a stage is too small to be a meaningful update, raise its max_tokens "
                "or merge stages. To override anyway, set "
                "i_understand_multi_epoch_inflates_forgetting: true."
            )
        return self

    def tokens_per_optimizer_step(self, seq_len: int, world_size: int) -> int:
        return (
            self.per_device_train_batch_size
            * self.gradient_accumulation_steps
            * world_size
            * seq_len
        )


class SFTTrainConfig(_Base):
    """``configs/train/sft_base.yaml`` — stub. See ``continual_llms.train.sft``."""

    cpt_checkpoint: str = "${CL_RUNS_DIR}/latest/final"
    base_model_for_tokenizer_check: str = "${CL_MODEL_DIR}"
    dataset: str | None = None
    chat_template: str | None = None
    completion_only_loss: bool = True
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    learning_rate: float = 1.0e-5
    num_train_epochs: float = 2.0
    max_seq_length: int = 4096
    bf16: bool = True
    gradient_checkpointing: bool = True


# --------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------


def expand_env_in_obj(obj: Any, table: dict[str, str] | None = None) -> Any:
    """Recursively expand ``${VAR}`` in every string of a parsed YAML tree."""
    table = table if table is not None else dict(os.environ)
    if isinstance(obj, str):
        return expand(obj, table)
    if isinstance(obj, dict):
        return {k: expand_env_in_obj(v, table) for k, v in obj.items()}
    if isinstance(obj, list):
        return [expand_env_in_obj(v, table) for v in obj]
    return obj


def load_config[T: BaseModel](path: str | Path, schema: type[T]) -> T:
    """Read a YAML file, expand ``${VAR}`` references, and validate against *schema*."""
    load_env()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise TypeError(f"{path} must contain a YAML mapping, got {type(raw).__name__}")
    return schema.model_validate(expand_env_in_obj(raw))