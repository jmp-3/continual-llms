"""The wandb tracking layer, driven without a GPU, a wandb account, or a Trainer.

The point of the key-set assertions is that the callback's contract is *what it emits*.
A dashboard that silently loses a series is indistinguishable from a run that was slow, so
the exact set is pinned here rather than eyeballed on the web UI.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest

from continual_llms.tracking.wandb_perf import (
    PER_STEP_METRIC_KEYS,
    flops_per_token,
    make_wandb_perf_callback,
    peak_flops_per_second,
    read_stage_runs,
    wandb_settings_from_env,
)

# Qwen3-4B-Base, the only model this repo trains.
QWEN3_4B = dict(n_parameters=4_022_468_096, n_layers=36, hidden_size=2560, seq_len=4096)


class FakeRun:
    """Enough of ``wandb.sdk.wandb_run.Run`` for the callback, and nothing more."""

    def __init__(self):
        self.logged: list[dict] = []
        self.defined: list[tuple[str, str | None]] = []
        self.summary: dict = {}
        self.config = SimpleNamespace(update=lambda values, **kw: self.config_updates.append(values))
        self.config_updates: list[dict] = []
        self.step = 0
        self.id = "fake-run"
        self.url = ""

    def log(self, payload, step=None):
        self.logged.append(payload)
        self.step = self.step + 1 if step is None else step

    def define_metric(self, name, step_metric=None, **kwargs):
        self.defined.append((name, step_metric))


def _stub_state(global_step: int, *, main: bool = True):
    return SimpleNamespace(global_step=global_step, is_world_process_zero=main)


def _stub_args(accum: int = 32):
    return SimpleNamespace(
        output_dir="/tmp/stage", logging_steps=5, gradient_accumulation_steps=accum
    )


def _run_steps(cb, n_steps: int, *, accum: int = 32, last_micro_batches: int | None = None):
    """Drive n optimizer steps, optionally ending on a partial accumulation."""
    args, control = _stub_args(accum), SimpleNamespace()
    cb.on_train_begin(args, _stub_state(0), control)
    for step in range(1, n_steps + 1):
        micro = accum if (step < n_steps or last_micro_batches is None) else last_micro_batches
        for _ in range(micro - 1):  # on_substep_end skips the batch that closes the step
            cb.on_substep_end(args, _stub_state(step), control)
        cb.on_step_end(args, _stub_state(step), control)


def _callback(run, **overrides):
    from transformers import TrainerCallback

    kwargs = dict(
        run=run,
        tokens_per_step=262_144,
        world_size=2,
        gradient_checkpointing=True,
        device_name="NVIDIA RTX A6000",
        log=lambda *_a, **_k: None,
        **QWEN3_4B,
    )
    kwargs.update(overrides)
    return make_wandb_perf_callback(TrainerCallback)(**kwargs)


# --------------------------------------------------------------------------------------
# the emitted key set
# --------------------------------------------------------------------------------------


def test_callback_emits_exactly_the_documented_key_set():
    run = FakeRun()
    _run_steps(_callback(run), 2)

    assert len(run.logged) == 2
    for payload in run.logged:
        assert set(payload) == set(PER_STEP_METRIC_KEYS)


def test_tokens_seen_is_cumulative_and_exact_from_the_micro_batch_count():
    """No cross-rank reduction is involved, so this must be exact, not approximate.

    ``tokens_per_optimizer_step`` already multiplies by world size and packed blocks are
    fixed-length with no padding, which is why an ``all_reduce`` on the logging path - the
    one thing most likely to hang a ZeRO-3 run - is unnecessary here.
    """
    run = FakeRun()
    _run_steps(_callback(run), 3)

    seen = [p["progress/total_tokens_seen"] for p in run.logged]
    assert seen == [262_144, 524_288, 786_432]
    assert [p["train/global_step"] for p in run.logged] == [1, 2, 3]


def test_a_partial_final_accumulation_is_not_over_credited():
    """The regression this guards: a short last step reported MFU above 100%.

    ``max_tokens`` rarely lands on a whole number of optimizer steps, so the last step of a
    stage usually closes early. Crediting it a full ``tokens_per_step`` divides real tokens
    by a short wall time; the observed value on the debug curriculum was ``perf/mfu = 2.06``.
    """
    run = FakeRun()
    _run_steps(_callback(run), 3, accum=32, last_micro_batches=4)

    # 4 of 32 micro-batches, i.e. 4 x (262144 / 32) tokens, not a whole step's worth.
    seen = [p["progress/total_tokens_seen"] for p in run.logged]
    assert seen == [262_144, 524_288, 524_288 + 4 * 8192]

    # Throughput scales with the step's real token count. Driven through step_metrics with
    # an explicit wall time, because the stub above runs steps in ~0 seconds.
    cb = _callback(FakeRun())
    full = cb.step_metrics(1, 90.0, 262_144)
    partial = cb.step_metrics(2, 90.0 / 8, 262_144 // 8)
    assert partial["perf/tokens_per_sec"] == pytest.approx(full["perf/tokens_per_sec"])
    assert partial["perf/mfu"] == pytest.approx(full["perf/mfu"])
    assert 0.0 < full["perf/mfu"] < 1.0


def test_perf_and_progress_series_are_registered_against_tokens_seen():
    run = FakeRun()
    _run_steps(_callback(run), 0)

    defined = dict(run.defined)
    assert defined["perf/*"] == "progress/total_tokens_seen"
    assert defined["progress/*"] == "progress/total_tokens_seen"


def test_num_parameters_is_overwritten_with_the_pre_engine_count():
    """transformers writes ``model.num_parameters()`` at on_train_begin, i.e. a ZeRO-3 shard."""
    run = FakeRun()
    _run_steps(_callback(run), 0)

    merged = {k: v for update in run.config_updates for k, v in update.items()}
    assert merged["model/num_parameters"] == QWEN3_4B["n_parameters"]


# --------------------------------------------------------------------------------------
# rank discipline
# --------------------------------------------------------------------------------------


def test_nothing_is_logged_off_rank_zero():
    """Invariant: logging is rank-0 only. Collectives, of which there are none here, are not."""
    run = FakeRun()
    cb = _callback(run)
    cb.on_train_begin(_stub_args(), _stub_state(0, main=False), SimpleNamespace())
    cb.on_step_end(_stub_args(), _stub_state(1, main=False), SimpleNamespace())

    assert run.logged == []
    assert run.defined == []
    assert run.config_updates == []


def test_a_none_run_is_a_complete_no_op():
    cb = _callback(None)
    control = SimpleNamespace(should_training_stop=False)
    assert cb.on_train_begin(_stub_args(), _stub_state(0), control) is control
    assert cb.on_step_end(_stub_args(), _stub_state(1), control) is control


# --------------------------------------------------------------------------------------
# MFU
# --------------------------------------------------------------------------------------


def test_flops_per_token_matches_the_hand_computed_qwen3_4b_figure():
    # 6N = 2.413e10, plus 12 * 36 * 2560 * 4096 = 4.53e9 for attention.
    assert flops_per_token(**QWEN3_4B) == pytest.approx(6 * 4_022_468_096 + 4_529_848_320)


def test_mfu_excludes_recompute_and_hfu_includes_it():
    """The 4/3 is gradient checkpointing's extra forward pass, and it must stay visible.

    MFU excluding recompute is what the ``assumed_tokens_per_second`` arithmetic in
    configs/train/cpt_base.yaml is expressed in, so folding recompute into it would make the
    logged number silently incomparable to the number the run was budgeted against.
    """
    cb = _callback(FakeRun())
    mfu, hfu = cb._mfu(1200.0)

    expected = 1200.0 * flops_per_token(**QWEN3_4B) / 154.8e12
    assert mfu == pytest.approx(expected)
    assert hfu == pytest.approx(expected * 4 / 3)
    assert 0.0 < mfu < 1.0

    no_ckpt = _callback(FakeRun(), gradient_checkpointing=False)
    assert no_ckpt._mfu(1200.0) == pytest.approx((expected, expected))


def test_an_unknown_device_yields_nan_and_warns_once():
    warnings: list[str] = []
    cb = _callback(FakeRun(), device_name="Some Future GPU", log=warnings.append)

    for _ in range(3):
        mfu, hfu = cb._mfu(1200.0)
        assert math.isnan(mfu) and math.isnan(hfu)
    assert len(warnings) == 1

    payload = cb.step_metrics(1, 1.0, 262_144)
    assert set(payload) == set(PER_STEP_METRIC_KEYS)
    assert math.isnan(payload["perf/mfu"])


def test_peak_flops_lookup_tolerates_the_nvidia_prefix():
    assert peak_flops_per_second("NVIDIA RTX A6000") == 154.8e12
    assert peak_flops_per_second("RTX A6000") == 154.8e12
    assert math.isnan(peak_flops_per_second(None))


# --------------------------------------------------------------------------------------
# env configuration and the stage-run index
# --------------------------------------------------------------------------------------


def test_mode_defaults_to_offline_and_env_wins(monkeypatch):
    monkeypatch.delenv("WANDB_MODE", raising=False)
    monkeypatch.delenv("WANDB_ENTITY", raising=False)
    monkeypatch.setenv("WANDB_PROJECT", "continual-llms")
    assert wandb_settings_from_env()["mode"] == "offline"

    monkeypatch.setenv("WANDB_MODE", "ONLINE")
    assert wandb_settings_from_env()["mode"] == "online"


def test_init_is_a_no_op_when_disabled(monkeypatch, tmp_path):
    from continual_llms.tracking.wandb_perf import init_stage_run

    monkeypatch.setenv("WANDB_MODE", "disabled")
    run = init_stage_run(
        run_id="r", stage_name="s01", stage_index=1, config={}, run_dir=tmp_path,
        log=lambda *_a: None,
    )
    assert run is None
    assert not (tmp_path / "wandb_runs.jsonl").exists()


def test_read_stage_runs_is_empty_without_the_file(tmp_path):
    assert read_stage_runs(tmp_path) == []


def test_read_stage_runs_keeps_the_latest_row_per_stage(tmp_path):
    """A stage restarted with --only-stage appends a second row for the same stage."""
    (tmp_path / "wandb_runs.jsonl").write_text(
        '{"stage": "s01", "stage_index": 1, "wandb_run_id": "old"}\n'
        '{"stage": "s02", "stage_index": 2, "wandb_run_id": "b"}\n'
        '{"stage": "s01", "stage_index": 1, "wandb_run_id": "new"}\n',
        encoding="utf-8",
    )
    rows = read_stage_runs(tmp_path)
    assert [r["wandb_run_id"] for r in rows] == ["new", "b"]


def test_module_imports_nothing_heavy_at_module_scope():
    """``load_env()`` must precede ``import torch``, and this module must be removable.

    A module-level ``import torch`` here would run before ``load_env()`` in any entry point
    that touches tracking, and CUDA_VISIBLE_DEVICES is read once at CUDA init - so the
    breakage would be silent, not loud. A module-level ``import wandb`` would additionally
    make the dependency mandatory.
    """
    import ast
    import pathlib

    import continual_llms.tracking.wandb_perf as module

    tree = ast.parse(pathlib.Path(module.__file__).read_text(encoding="utf-8"))
    heavy = {"torch", "wandb", "transformers", "deepspeed", "accelerate", "numpy"}
    for node in tree.body:  # module scope only; imports inside functions are the convention
        if isinstance(node, ast.Import):
            assert not {n.name.split(".")[0] for n in node.names} & heavy, ast.dump(node)
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            assert (node.module or "").split(".")[0] not in heavy, ast.dump(node)


def test_init_fails_open_when_wandb_cannot_be_imported(monkeypatch, tmp_path):
    """A missing or broken wandb must cost a warning, not a stage."""
    import builtins

    from continual_llms.tracking.wandb_perf import init_stage_run

    monkeypatch.setenv("WANDB_MODE", "offline")
    real_import = builtins.__import__

    def no_wandb(name, *args, **kwargs):
        if name == "wandb" or name.startswith("wandb."):
            raise ModuleNotFoundError("No module named 'wandb'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_wandb)
    warnings: list[str] = []
    run = init_stage_run(
        run_id="r", stage_name="s01", stage_index=1, config={}, run_dir=tmp_path,
        log=warnings.append,
    )
    assert run is None
    assert len(warnings) == 1 and "wandb" in warnings[0]
    assert not (tmp_path / "wandb_runs.jsonl").exists()


def test_matrix_wandb_payload_is_namespaced_and_drops_missing_metrics():
    from continual_llms.eval.matrix import wandb_summary_payload

    payload = wandb_summary_payload(
        {
            "average_forgetting_nll": 0.02,
            "backward_transfer_nll": 0.01,
            "forward_transfer_nll": None,
            "ppl_ratios": {"average_forgetting": 1.02, "forward_transfer": None},
        }
    )
    assert payload == {
        "cl/average_forgetting_nll": 0.02,
        "cl/backward_transfer_nll": 0.01,
        "cl/ppl_ratio/average_forgetting": 1.02,
    }
    assert all(key.startswith("cl/") for key in payload)
