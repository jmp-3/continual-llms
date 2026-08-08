"""Run tracking. Observability only - nothing here may change what training computes.

The one module, :mod:`continual_llms.tracking.wandb_perf`, is deliberately importable with
``wandb`` absent and imports nothing heavy at module scope, so removing the dependency (or
this package) cannot break the trainer.
"""

from __future__ import annotations

__all__: list[str] = []
