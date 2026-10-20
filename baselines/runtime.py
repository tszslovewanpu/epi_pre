"""Resolve machine-local storage separately from versioned experiment settings.

Precedence: environment > machine.local.json > repository-local defaults.
Nothing in this module creates directories, downloads data or imports torch.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from string import Template
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULTS = {
    "data_root": "dataset/asepv1.2.0",
    "cache_root": ".local/cache",
    "runs_root": ".local/runs",
}
_ENV = {key: f"EPITOPE_{key.upper()}" for key in _DEFAULTS}
_PATH_KEYS = {
    "id_list_path", "interim_graph_dir", "pdb_root", "split_path",
    "cache_path", "deepinteract_context_cache", "output_dir",
}


def machine_paths() -> dict[str, Path]:
    """Read private per-machine paths; relative values are repository-relative."""
    local_path = REPO_ROOT / "machine.local.json"
    local = json.loads(local_path.read_text(encoding="utf-8")) if local_path.exists() else {}
    if not isinstance(local, dict) or set(local) - set(_DEFAULTS):
        raise ValueError(f"{local_path} must map only: {', '.join(_DEFAULTS)}")
    resolved = {}
    for key, default in _DEFAULTS.items():
        raw = os.environ.get(_ENV[key], local.get(key, default))
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{key} must be a nonempty path string")
        value = Path(os.path.expandvars(raw)).expanduser()
        if "$" in str(value):
            raise ValueError(f"Unresolved environment variable in {key}: {value}")
        resolved[key] = (value if value.is_absolute() else REPO_ROOT / value).resolve()
    return resolved


def resolve_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return a copy with storage placeholders and path fields resolved."""
    paths = machine_paths()
    variables = {**os.environ, "EPITOPE_REPO_ROOT": str(REPO_ROOT),
                 **{_ENV[key]: str(value) for key, value in paths.items()}}

    def resolve(value: Any, key: str = "") -> Any:
        if isinstance(value, dict):
            return {k: resolve(v, k) for k, v in value.items()}
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, str):
            if "$" in value:
                try:
                    value = Template(value).substitute(variables)
                except (KeyError, ValueError) as error:
                    raise ValueError(f"Invalid or undefined variable in config field {key!r}") from error
            if key in _PATH_KEYS and value:
                path = Path(value).expanduser()
                value = str((path if path.is_absolute() else REPO_ROOT / path).resolve())
        return value

    return resolve(copy.deepcopy(config))
