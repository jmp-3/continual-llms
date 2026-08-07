"""Project ``.env`` loading.

Import and call :func:`load_env` as the *first* statement of every entry point, before
``torch`` (or anything that imports it) is loaded. ``CUDA_VISIBLE_DEVICES`` is only read
once, at CUDA initialisation, so setting it after ``import torch`` is silently ineffective.

This module deliberately imports nothing heavy.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

__all__ = ["load_env", "find_dotenv", "cpt_root", "env_path", "expand"]

_VAR = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)")


def find_dotenv(start: Path | str | None = None) -> Path | None:
    """Walk upwards from *start* looking for a ``.env``. Honours ``$CPT_ROOT`` first."""
    root = os.environ.get("CPT_ROOT")
    if root and (Path(root) / ".env").is_file():
        return Path(root) / ".env"
    here = Path(start) if start is not None else Path(__file__).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / ".env").is_file():
            return candidate / ".env"
    return None


def expand(value: str, table: dict[str, str]) -> str:
    """Expand ``${VAR}``/``$VAR`` against *table* then the live environment."""

    def sub(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return table.get(name, os.environ.get(name, ""))

    return _VAR.sub(sub, value)


def load_env(path: Path | str | None = None, *, override: bool = False) -> dict[str, str]:
    """Parse the project ``.env`` into ``os.environ``.

    Existing environment variables win unless *override* is set, so an explicit
    ``CUDA_VISIBLE_DEVICES=0,4 uv run ...`` on the command line still takes effect.

    Returns the parsed mapping (fully expanded), even for keys that were not applied.
    """
    dotenv = Path(path) if path is not None else find_dotenv()
    parsed: dict[str, str] = {}
    if dotenv is None or not dotenv.is_file():
        return parsed

    for raw in dotenv.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        parsed[key] = expand(value, parsed)

    for key, value in parsed.items():
        if override or key not in os.environ:
            os.environ[key] = value
    return parsed


def env_path(name: str, default: str | None = None) -> Path:
    """Read *name* from the environment as a :class:`Path`, raising if unset."""
    value = os.environ.get(name, default)
    if value is None:
        raise KeyError(
            f"{name} is not set. Source the project .env, or call cpt_m2d2.env.load_env() "
            "before reading configuration."
        )
    return Path(value)


def cpt_root() -> Path:
    load_env()
    return env_path("CPT_ROOT")