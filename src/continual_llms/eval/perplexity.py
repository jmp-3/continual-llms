"""Token-level held-out NLL / perplexity per domain.

One row of the matrix is ``len(eval_domains)`` evaluations, and the matrix is
``O(stages x domains)`` rows, so this is written to be cheap and exactly reproducible
rather than flexible:

* blocks are pre-packed and all exactly ``seq_len`` long, so there is no padding and no
  masking to get wrong;
* NLL is summed with ``reduction="sum"`` in float32 over sequence chunks, never averaged
  per batch and re-averaged, so the reported number is a true token-level mean;
* every rank processes the *same number of batches* (short ranks re-run a dummy batch that
  is weighted out), which keeps DeepSpeed's collectives in lockstep. Under ZeRO-3 a rank
  that skips a forward pass deadlocks the others.
"""

from __future__ import annotations

import argparse
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from ..env import load_env

__all__ = ["PerplexityResult", "evaluate_blocks", "evaluate_domains", "main"]


@dataclass
class PerplexityResult:
    domain: str
    nll: float
    ppl: float
    n_tokens: int
    n_blocks: int
    seq_len: int
    seconds: float

    def as_row(self, **extra) -> dict:
        return {**asdict(self), **extra}


def _distributed():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return dist, dist.get_rank(), dist.get_world_size()
    return None, 0, 1


def _sum_nll(logits, labels, *, chunk: int = 512):
    """Sum of token NLL over a batch, computed in float32 over sequence chunks.

    Chunking matters: a single ``(2, 4095, 151936)`` float32 upcast is ~5 GB, which is more
    headroom than remains after ZeRO-2's optimizer state. Chunked, the temporary is bounded
    by ``chunk x vocab_size``.
    """
    import torch
    import torch.nn.functional as F

    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    total = torch.zeros((), dtype=torch.float32, device=logits.device)
    n = 0
    seq = shift_logits.shape[1]
    for start in range(0, seq, chunk):
        stop = min(start + chunk, seq)
        piece = shift_logits[:, start:stop, :].reshape(-1, shift_logits.shape[-1]).float()
        target = shift_labels[:, start:stop].reshape(-1)
        total += F.cross_entropy(piece, target, reduction="sum")
        n += int(target.numel())
    return total, n


def evaluate_blocks(
    model,
    blocks,
    *,
    device,
    batch_size: int = 2,
    domain: str | None = None,
    chunk: int = 512,
) -> PerplexityResult:
    """Mean token NLL of *model* over every block of *blocks*.

    ``blocks`` is a :class:`continual_llms.data.loader.PackedBlockArray`. *model* may be a bare
    ``nn.Module`` or a DeepSpeed engine; under ZeRO-3 the engine's own forward hooks gather
    parameters per layer, so no explicit gather is needed here.
    """
    import numpy as np
    import torch

    dist, rank, world = _distributed()
    started = time.time()

    n_blocks = blocks.n_blocks
    if n_blocks == 0:
        raise ValueError(f"{blocks.bin_path} has no blocks")

    mine = list(range(rank, n_blocks, world))
    per_rank = math.ceil(n_blocks / world)
    # Pad with block 0 at weight 0 so every rank runs the same number of forward passes.
    weights = [1.0] * len(mine) + [0.0] * (per_rank - len(mine))
    mine = mine + [0] * (per_rank - len(mine))
    n_batches = math.ceil(per_rank / batch_size)

    was_training = model.training
    model.eval()
    nll_sum = torch.zeros((), dtype=torch.float64, device=device)
    tok_sum = torch.zeros((), dtype=torch.float64, device=device)

    with torch.no_grad():
        for b in range(n_batches):
            lo, hi = b * batch_size, min((b + 1) * batch_size, per_rank)
            idx = mine[lo:hi]
            w = weights[lo:hi]
            batch = np.stack([blocks.block(i).astype(np.int64) for i in idx])
            input_ids = torch.from_numpy(batch).to(device, non_blocking=True)
            out = model(input_ids=input_ids, use_cache=False)
            logits = out.logits if hasattr(out, "logits") else out[0]
            total, n = _sum_nll(logits, input_ids, chunk=chunk)
            # Rows with weight 0 are padding; drop their contribution without skipping the
            # forward pass that keeps the collectives aligned.
            if all(x == 1.0 for x in w):
                nll_sum += total.double()
                tok_sum += float(n)
            else:
                for j, weight in enumerate(w):
                    if weight:
                        row_total, row_n = _sum_nll(
                            logits[j : j + 1], input_ids[j : j + 1], chunk=chunk
                        )
                        nll_sum += row_total.double()
                        tok_sum += float(row_n)
            del logits, out

    if dist is not None:
        stacked = torch.stack([nll_sum, tok_sum])
        dist.all_reduce(stacked, op=dist.ReduceOp.SUM)
        nll_sum, tok_sum = stacked[0], stacked[1]

    if was_training:
        model.train()

    n_tokens = int(tok_sum.item())
    mean_nll = float(nll_sum.item()) / max(1, n_tokens)
    return PerplexityResult(
        domain=domain or blocks.bin_path.parent.name,
        nll=mean_nll,
        ppl=float(math.exp(min(mean_nll, 700.0))),
        n_tokens=n_tokens,
        n_blocks=n_blocks,
        seq_len=blocks.seq_len,
        seconds=round(time.time() - started, 1),
    )


def evaluate_domains(
    model,
    *,
    packed_dir: str | Path,
    domains: list[str],
    seq_len: int,
    device,
    batch_size: int = 2,
    max_tokens: int | None = None,
    log=print,
) -> list[PerplexityResult]:
    from ..data.loader import load_eval_blocks

    results: list[PerplexityResult] = []
    for domain in domains:
        blocks = load_eval_blocks(
            packed_dir=packed_dir, domain=domain, seq_len=seq_len, max_tokens=max_tokens
        )
        result = evaluate_blocks(
            model, blocks, device=device, batch_size=batch_size, domain=domain
        )
        log(
            f"  eval {domain:<48} nll={result.nll:.4f}  ppl={result.ppl:8.3f}  "
            f"({result.n_tokens:,} tokens, {result.seconds}s)"
        )
        results.append(result)
    return results


# --------------------------------------------------------------------------------------
# standalone CLI (used for the baseline row, and for re-scoring a saved checkpoint)
# --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    load_env()
    root = Path(os.environ.get("CL_ROOT", Path(__file__).resolve().parents[3]))

    parser = argparse.ArgumentParser(prog="continual_llms.eval.perplexity")
    parser.add_argument("--model", required=True, help="checkpoint directory to score")
    parser.add_argument("--data-config", type=Path, default=root / "configs/data/m2d2_stages.yaml")
    parser.add_argument("--model-config", type=Path, default=root / "configs/model/qwen3_4b.yaml")
    parser.add_argument("--domains", nargs="*", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--out", type=Path, default=None, help="append rows to this jsonl")
    parser.add_argument("--stage-index", type=int, default=0)
    parser.add_argument("--stage-name", default="base")
    parser.add_argument("--run-id", default="standalone")
    args = parser.parse_args(argv)

    import torch

    from ..config import DataConfig, ModelConfig, load_config
    from ..data.loader import discover_packed_domains
    from ..train.common import append_jsonl_fsync, is_main_process, load_causal_lm, setup_logging

    logger = setup_logging()
    data_cfg = load_config(args.data_config, DataConfig)
    model_cfg = load_config(args.model_config, ModelConfig)

    domains = args.domains or (
        discover_packed_domains(data_cfg.packed_dir)
        if data_cfg.eval_domains == ["ALL"]
        else data_cfg.eval_domains
    )
    if not domains:
        logger.error("no packed eval domains found under %s", data_cfg.packed_dir)
        return 2

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model = load_causal_lm(args.model, model_cfg).to(device)

    results = evaluate_domains(
        model,
        packed_dir=data_cfg.packed_dir,
        domains=domains,
        seq_len=data_cfg.seq_len,
        device=device,
        batch_size=args.batch_size,
        max_tokens=data_cfg.eval_max_tokens,
        log=logger.info,
    )

    if args.out and is_main_process():
        append_jsonl_fsync(
            args.out,
            [
                r.as_row(
                    run_id=args.run_id,
                    stage_index=args.stage_index,
                    stage_name=args.stage_name,
                    checkpoint=str(args.model),
                    timestamp=time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                )
                for r in results
            ],
        )
        logger.info("wrote %d rows to %s", len(results), args.out)

    # Tracking. This CLI produces row 0 of the matrix - the untouched base model that
    # forward transfer is measured against - so it gets a run in the same group as the
    # stages, with job_type "base". Fail-open, rank 0 only, and a no-op when wandb is off.
    from ..tracking.wandb_perf import (
        environment_config,
        finish_run,
        init_stage_run,
        log_eval_results,
    )

    run_dir = Path(os.environ.get("CL_RUNS_DIR", root / "runs")) / args.run_id
    wandb_run = init_stage_run(
        run_id=args.run_id,
        stage_name=args.stage_name,
        stage_index=args.stage_index,
        config={
            **environment_config(
                root=root,
                seq_len=data_cfg.seq_len,
                per_device_train_batch_size=0,
                gradient_accumulation_steps=0,
                world_size=1,
                tokens_per_optimizer_step=0,
                base_model_dir=args.model,
            ),
            "stage/name": args.stage_name,
            "stage/index": args.stage_index,
            "stage/domain": "",
            "stage/model_source": str(args.model),
            "eval_domains": domains,
            "eval_max_tokens": data_cfg.eval_max_tokens,
        },
        run_dir=run_dir if run_dir.is_dir() else None,
        log=logger.warning,
    )
    log_eval_results(wandb_run, results, global_step=0, log=logger.warning)
    finish_run(wandb_run, log=logger.warning)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())