"""Packed ``uint32`` memmaps -> torch datasets, with replay mixing.

Nothing here re-tokenizes: a packed ``.bin`` is already ``n_blocks * seq_len`` token ids, so
a block is a slice and ``__getitem__`` is a memmap read plus one dtype cast.

Replay is expressed as a fraction of a stage's **total** training tokens, drawn from a domain
that is never itself a training stage. Blocks are sampled without replacement within a stage,
and the realized fraction is returned so the caller can log it and notice drift.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .tokenize_pack import PACKED_DTYPE

__all__ = [
    "PackedMeta",
    "PackedBlockArray",
    "BlockDataset",
    "PackedCollator",
    "StageDataStats",
    "build_stage_dataset",
    "load_eval_blocks",
]


@dataclass(frozen=True)
class PackedMeta:
    domain: str
    kind: str
    seq_len: int
    dtype: str
    n_tokens: int
    n_blocks: int
    tokenizer_hash: str
    eos_token_id: int

    @classmethod
    def load(cls, meta_path: Path) -> "PackedMeta":
        raw = json.loads(Path(meta_path).read_text())
        return cls(
            domain=raw["domain"],
            kind=raw["kind"],
            seq_len=raw["seq_len"],
            dtype=raw["dtype"],
            n_tokens=raw["n_tokens"],
            n_blocks=raw["n_blocks"],
            tokenizer_hash=raw["tokenizer_hash"],
            eos_token_id=raw["eos_token_id"],
        )


class PackedBlockArray:
    """Read-only view of ``<domain>/<kind>.bin`` as ``n_blocks x seq_len``.

    The memmap is opened lazily so the object survives being pickled into dataloader
    workers.
    """

    def __init__(self, bin_path: str | Path, seq_len: int, *, meta: PackedMeta | None = None):
        self.bin_path = Path(bin_path)
        self.seq_len = seq_len
        if not self.bin_path.is_file():
            raise FileNotFoundError(
                f"{self.bin_path} does not exist. Run "
                "`uv run python -m cpt_m2d2.data.tokenize_pack` first."
            )
        meta_path = self.bin_path.with_suffix(".meta.json")
        self.meta = meta or (PackedMeta.load(meta_path) if meta_path.is_file() else None)
        if self.meta is not None:
            if self.meta.dtype != PACKED_DTYPE:
                raise ValueError(f"{meta_path}: dtype {self.meta.dtype!r}, expected {PACKED_DTYPE!r}")
            if self.meta.seq_len != seq_len:
                raise ValueError(
                    f"{self.bin_path} was packed at seq_len={self.meta.seq_len} but the config "
                    f"asks for {seq_len}. Repack with --force."
                )
        size = self.bin_path.stat().st_size
        itemsize = np.dtype(PACKED_DTYPE).itemsize
        if size % (seq_len * itemsize):
            raise ValueError(
                f"{self.bin_path} is {size} bytes, not a whole number of "
                f"{seq_len}-token {PACKED_DTYPE} blocks"
            )
        self.n_blocks = size // (seq_len * itemsize)
        self._mm: np.memmap | None = None

    @property
    def n_tokens(self) -> int:
        return self.n_blocks * self.seq_len

    def _memmap(self) -> np.memmap:
        if self._mm is None:
            self._mm = np.memmap(
                self.bin_path, dtype=PACKED_DTYPE, mode="r", shape=(self.n_blocks, self.seq_len)
            )
        return self._mm

    def block(self, index: int) -> np.ndarray:
        return np.asarray(self._memmap()[index])

    def __len__(self) -> int:
        return self.n_blocks

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_mm"] = None  # memmaps do not survive pickling to a worker
        return state

    def __repr__(self) -> str:
        return f"PackedBlockArray({self.bin_path.name}, {self.n_blocks:,} blocks)"


class BlockDataset(Dataset):
    """A plan over one or more :class:`PackedBlockArray`s.

    ``index`` is an ``(N, 2)`` int array of ``(source_id, block_id)``. Keeping the plan
    explicit is what makes replay mixing auditable: the realized fraction is a property of
    this array, not of a sampler somewhere downstream.
    """

    def __init__(self, sources: list[PackedBlockArray], index: np.ndarray, *, source_names: list[str] | None = None):
        if index.ndim != 2 or index.shape[1] != 2:
            raise ValueError(f"index must be (N, 2), got {index.shape}")
        self.sources = sources
        self.index = index.astype(np.int64, copy=False)
        self.source_names = source_names or [s.bin_path.parent.name for s in sources]
        self.seq_len = sources[0].seq_len

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        source_id, block_id = self.index[i]
        block = self.sources[int(source_id)].block(int(block_id))
        # uint32 has no torch equivalent that nn.Embedding accepts; int64 is what the
        # embedding lookup wants anyway.
        ids = torch.from_numpy(block.astype(np.int64, copy=True))
        return {"input_ids": ids, "labels": ids.clone(), "source_id": int(source_id)}

    def source_token_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {name: 0 for name in self.source_names}
        ids, tallies = np.unique(self.index[:, 0], return_counts=True)
        for source_id, tally in zip(ids, tallies):
            counts[self.source_names[int(source_id)]] = int(tally) * self.seq_len
        return counts


class PackedCollator:
    """Stack pre-packed blocks. No padding is possible: every block is exactly seq_len."""

    def __call__(self, features: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        input_ids = torch.stack([f["input_ids"] for f in features])
        labels = torch.stack([f["labels"] for f in features])
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        }


@dataclass
class StageDataStats:
    domain: str
    seq_len: int
    main_blocks: int
    main_tokens: int
    main_blocks_available: int
    replay_source: str | None
    replay_blocks: int
    replay_tokens: int
    replay_pool_blocks: int
    target_replay_ratio: float
    realized_replay_ratio: float
    total_tokens: int
    capped_by_max_tokens: bool

    def as_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        parts = [
            f"stage data: {self.domain}",
            f"  main   : {self.main_blocks:,} blocks / {self.main_tokens:,} tokens"
            + (f"  (capped from {self.main_blocks_available:,} blocks)" if self.capped_by_max_tokens else ""),
        ]
        if self.replay_source:
            parts.append(
                f"  replay : {self.replay_blocks:,} blocks / {self.replay_tokens:,} tokens "
                f"from {self.replay_source} (pool {self.replay_pool_blocks:,} blocks)"
            )
            parts.append(
                f"  ratio  : target {self.target_replay_ratio:.4f} -> realized "
                f"{self.realized_replay_ratio:.4f}"
            )
        parts.append(f"  total  : {self.total_tokens:,} tokens")
        return "\n".join(parts)


def _blocks_for(packed_dir: Path, domain: str, kind: str, seq_len: int) -> PackedBlockArray:
    return PackedBlockArray(Path(packed_dir) / domain / f"{kind}.bin", seq_len)


def build_stage_dataset(
    *,
    packed_dir: str | Path,
    domain: str,
    seq_len: int,
    max_tokens: int | None = None,
    replay_source: str | None = None,
    replay_ratio: float = 0.0,
    seed: int = 1234,
    strict_ratio: bool = False,
    ratio_tolerance: float = 0.005,
    sampling: str = "head",
) -> tuple[BlockDataset, StageDataStats]:
    """Assemble one stage's training set: the domain's blocks plus mixed-in replay.

    ``max_tokens`` caps the *main* domain only, and is applied here rather than at pack time
    so packed artifacts stay curriculum-independent: permuting the stage order or changing a
    cap invalidates no ``.bin``. ``sampling`` decides which blocks the cap keeps - ``"head"``
    (default, deterministic, sequential reads) or ``"random"`` (without replacement).

    Replay is then sized so that replay tokens are ``replay_ratio`` of the combined total::

        n_replay = round(n_main * ratio / (1 - ratio))

    Replay blocks are drawn without replacement, so if the pool is smaller than the target
    the realized ratio falls short. That is reported (and, with ``strict_ratio``, fatal)
    rather than silently sampled with replacement, which would over-weight a handful of
    blocks and quietly change what the control measures.
    """
    packed_dir = Path(packed_dir)
    main = _blocks_for(packed_dir, domain, "train", seq_len)

    n_main = main.n_blocks
    capped = False
    if max_tokens is not None:
        allowed = max_tokens // seq_len
        if allowed < n_main:
            n_main, capped = allowed, True
    if n_main <= 0:
        raise ValueError(f"{domain}: max_tokens={max_tokens} leaves 0 blocks at seq_len={seq_len}")

    sources = [main]
    source_names = [domain]
    if capped and sampling == "random":
        main_blocks = np.random.default_rng(seed + 2).permutation(main.n_blocks)[:n_main]
        main_blocks.sort()  # keep the memmap reads roughly sequential
    else:
        main_blocks = np.arange(n_main, dtype=np.int64)
    index = np.stack([np.zeros(n_main, dtype=np.int64), main_blocks.astype(np.int64)], axis=1)

    n_replay = 0
    pool_blocks = 0
    if replay_source and replay_ratio > 0:
        if replay_source == domain:
            raise ValueError(
                f"replay.source == stage domain ({domain}); replay must come from a domain "
                "that is not being trained, otherwise it is not a control."
            )
        replay = _blocks_for(packed_dir, replay_source, "train", seq_len)
        pool_blocks = replay.n_blocks
        want = int(round(n_main * replay_ratio / (1.0 - replay_ratio)))
        n_replay = min(want, pool_blocks)
        if n_replay < want:
            message = (
                f"replay pool for {replay_source} has only {pool_blocks:,} blocks but "
                f"{want:,} were needed for ratio {replay_ratio}; realized ratio will be lower"
            )
            if strict_ratio:
                raise ValueError(message)
            print(f"WARNING: {message}")
        if n_replay > 0:
            rng = np.random.default_rng(seed)
            chosen = rng.permutation(pool_blocks)[:n_replay].astype(np.int64)
            sources.append(replay)
            source_names.append(replay_source)
            replay_index = np.stack([np.ones(n_replay, dtype=np.int64), chosen], axis=1)
            index = np.concatenate([index, replay_index], axis=0)

    # Deterministic interleave so replay is spread through the stage even if a caller
    # disables the sampler's own shuffling.
    np.random.default_rng(seed + 1).shuffle(index)

    total_blocks = n_main + n_replay
    realized = (n_replay / total_blocks) if total_blocks else 0.0
    if strict_ratio and abs(realized - replay_ratio) > ratio_tolerance:
        raise ValueError(
            f"realized replay ratio {realized:.4f} differs from target {replay_ratio:.4f} "
            f"by more than {ratio_tolerance}"
        )

    stats = StageDataStats(
        domain=domain,
        seq_len=seq_len,
        main_blocks=n_main,
        main_tokens=n_main * seq_len,
        main_blocks_available=main.n_blocks,
        replay_source=replay_source if n_replay else None,
        replay_blocks=n_replay,
        replay_tokens=n_replay * seq_len,
        replay_pool_blocks=pool_blocks,
        target_replay_ratio=replay_ratio,
        realized_replay_ratio=realized,
        total_tokens=total_blocks * seq_len,
        capped_by_max_tokens=capped,
    )
    return BlockDataset(sources, index, source_names=source_names), stats


def load_eval_blocks(
    *, packed_dir: str | Path, domain: str, seq_len: int, max_tokens: int | None = None
) -> PackedBlockArray:
    """Held-out blocks for one domain. ``max_tokens`` is normally already applied at pack
    time; re-applying here lets a run shrink the eval budget without repacking."""
    array = _blocks_for(Path(packed_dir), domain, "eval", seq_len)
    if max_tokens is not None and array.n_blocks * seq_len > max_tokens:
        array.n_blocks = max(1, max_tokens // seq_len)
    return array


def discover_packed_domains(packed_dir: str | Path, kind: str = "eval") -> list[str]:
    """Every domain with a packed ``<kind>.bin``, sorted. Backs ``eval_domains: [ALL]``."""
    packed_dir = Path(packed_dir)
    if not packed_dir.is_dir():
        return []
    return sorted(d.name for d in packed_dir.iterdir() if (d / f"{kind}.bin").is_file())