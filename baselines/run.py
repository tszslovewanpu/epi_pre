"""Portable research launcher.

Run from the repository root with ``python -m baselines.run --help``.
Local data and experiment artifacts are never part of the Git workflow.
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import datetime as dt
import fcntl
import hashlib
import importlib
import json
import platform
import re
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterator

import torch

from baselines import runtime
from baselines.se3bench import train as shared

METHODS = (
    "lf3drope", "gf3drope", "gf3drope_rotaug", "distance_rbf",
    "local_geometry", "ipa", "gvp", "geformer", "deepinteract", "1drope",
    "lf3drope_sparse_edge", "lf3drope_geometry_value", "lf3drope_bond_angle",
    "lf3drope_module_ablation",
)
_HOOKS = (
    "METHODS", "build_model", "implementation_fingerprint",
    "load_or_build_dataset", "run_epoch", "prepare_sample",
)


def implementation_fingerprint(method: str) -> tuple[str, list[str]]:
    """Include shared code and transitive model helpers in the portable hash."""
    root = Path(__file__).resolve().parent
    files = {Path("run.py"), Path("runtime.py")}
    # A broad source hash also captures helpers imported by model adapters.
    for directory in ("se3bench", *METHODS):
        for path in (root / directory).glob("*.py"):
            if not path.name.startswith("test_"):
                files.add(path.relative_to(root))
    digest = hashlib.sha256()
    names = sorted(path.as_posix() for path in files)
    for name in names:
        digest.update(name.encode() + b"\0" + (root / name).read_bytes() + b"\0")
    return digest.hexdigest(), names


def build_module_variant(
    method: str, input_ab_dim: int, input_ag_dim: int,
    hparams: dict[str, Any], device: torch.device,
) -> torch.nn.Module:
    """Build variants that expose an explicit cross-attention switch."""
    if method not in ("1drope", "lf3drope_module_ablation"):
        raise ValueError(f"unsupported module variant: {method!r}")
    model_class = importlib.import_module(f"baselines.{method}.model").Model
    return model_class(
        input_ab_dim=input_ab_dim,
        input_ag_dim=input_ag_dim,
        d_model=int(hparams.get("d_model", 256)),
        n_heads=int(hparams.get("n_heads", 8)),
        n_layers_ab=int(hparams.get("n_layers_ab", 4)),
        n_layers_ag=int(hparams.get("n_layers_ag", 4)),
        n_layers_cross=int(hparams.get("n_layers_cross", 2)),
        dim_ff=int(hparams.get("dim_ff", 512)),
        dropout=float(hparams.get("dropout", 0.1)),
        use_cdr_type=bool(hparams.get("use_cdr_type", False)),
        use_cross_attention=bool(hparams.get("use_cross_attention", True)),
        use_global_context=bool(hparams.get("use_global_context", True)),
        use_pair_mil=bool(hparams.get("use_pair_mil", True)),
        mil_weight=float(hparams.get("mil_weight", 0.75)),
        geometry_config=dict(hparams.get("geometry", {}) or {}),
    ).to(device)


@contextlib.contextmanager
def configured_trainer(method: str) -> Iterator[None]:
    """Install method adapters only for this invocation, then restore hooks."""
    if method not in METHODS:
        raise ValueError(f"unsupported method {method!r}; choose from {METHODS}")
    saved = {name: getattr(shared, name) for name in _HOOKS}
    try:
        shared.METHODS = METHODS
        if method == "deepinteract":
            importlib.import_module("baselines.deepinteract.train").configure_trainer()
        elif method in ("1drope", "lf3drope_module_ablation"):
            shared.build_model = build_module_variant
        shared.implementation_fingerprint = implementation_fingerprint
        yield
    finally:
        for name, value in saved.items():
            setattr(shared, name, value)


def prepare_config(config: dict[str, Any], run_name: str | None = None) -> dict[str, Any]:
    config = runtime.resolve_config(copy.deepcopy(config))
    if config.get("method") not in METHODS:
        raise ValueError(f"method must be one of {METHODS}")
    dataset = config.setdefault("dataset", {})
    split = config.get("split", str(dataset.get("split_method", "epitope_ratio")).removeprefix("epitope_"))
    if split not in ("ratio", "group"):
        raise ValueError("split must be ratio or group")
    expected_split = f"epitope_{split}"
    if dataset.get("split_method", expected_split) not in (split, expected_split):
        raise ValueError(f"split={split} requires dataset.split_method={expected_split}; override both fields together")
    config["split"], dataset["split_method"] = split, expected_split
    if run_name is not None:
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
            raise ValueError("run-name must start with a letter/digit and use only letters, digits, _, - or .")
        config["output_dir"] = str(runtime.machine_paths()["runs_root"] / run_name)
    if not config.get("output_dir"):
        raise ValueError("provide --run-name NAME or an explicit output_dir in the configuration")
    output = Path(config["output_dir"]).expanduser().resolve()
    # Repository source and local data roots are never experiment destinations.
    protected = [runtime.REPO_ROOT.resolve(), *(path.resolve() for path in runtime.machine_paths().values())]
    if output in protected:
        raise ValueError(f"output_dir must be a dedicated experiment subdirectory: {output}")
    config["output_dir"] = str(output)
    with configured_trainer(str(config["method"])):
        shared._validate_experiment_config(config)
    if config["method"] == "1drope" and config.get("hparams", {}).get("use_cross_attention") is False:
        raise ValueError("1drope requires hparams.use_cross_attention=true")
    return config


def _utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _git_info() -> dict[str, Any]:
    def git(*args: str) -> str:
        return subprocess.check_output(
            ["git", "-C", str(runtime.REPO_ROOT), *args],
            stderr=subprocess.DEVNULL, text=True,
        ).strip()
    try:
        commit = git("rev-parse", "HEAD")
    except (OSError, subprocess.CalledProcessError):
        commit = None
    try:
        changes = git("status", "--porcelain=v1", "--untracked-files=all")
        return {"commit": commit, "dirty": bool(changes), "changes": changes.splitlines()}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": commit, "dirty": None, "changes": [], "available": False}


def _input_digest(value: Any) -> dict[str, Any]:
    if not value:
        return {"path": None, "sha256": None}
    path = Path(value)
    if not path.is_file():
        return {"path": str(path), "sha256": None, "missing": True}
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest()}


def run_metadata(config: dict[str, Any], mode: str) -> dict[str, Any]:
    gpu = None
    if mode != "build-cache" and config.get("try_gpu", True):
        index = int(config.get("gpu", 0))
        properties = torch.cuda.get_device_properties(index)
        gpu = {"index": index, "name": properties.name, "total_memory_bytes": properties.total_memory}
    dataset = config.get("dataset", {})
    return {
        "started_at": _utc_now(), "status": "running", "mode": mode,
        "git": _git_info(), "host": socket.gethostname(), "platform": platform.platform(),
        "python": sys.version, "python_executable": sys.executable,
        "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": gpu,
        "command": [sys.executable, *sys.argv],
        "config": config,
        "input_files": {key: _input_digest(dataset.get(key)) for key in ("id_list_path", "split_path")},
        "implementation_fingerprint": implementation_fingerprint(str(config["method"]))[0],
    }


def scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    """Compare checkpoint protocols while allowing machine-local placement."""
    value = copy.deepcopy(config)
    for key in ("output_dir", "gpu", "try_gpu", "num_threads", "cuda_memory_fraction"):
        value.pop(key, None)
    dataset = value.get("dataset", {})
    for key in ("id_list_path", "interim_graph_dir", "pdb_root", "split_path", "cache_path",
                "deepinteract_context_cache", "rebuild_cache", "verbose"):
        dataset.pop(key, None)
    value.get("hparams", {}).pop("progress", None)
    return value


def validate_evaluation(config: dict[str, Any], output: Path) -> None:
    checkpoint_path = output / "best_model.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("config"), dict):
        raise ValueError("evaluation checkpoint must contain its original experiment configuration")
    if scientific_config(checkpoint["config"]) != scientific_config(config):
        raise ValueError("evaluation configuration differs from checkpoint science settings (split/seed/features/model/training); use the original run configuration")
    # New portable runs also retain path-independent identities of these files.
    metadata_path = output / "run_metadata.json"
    if metadata_path.is_file():
        original = json.loads(metadata_path.read_text()).get("input_files", {})
        for key in ("id_list_path", "split_path"):
            digest = original.get(key, {}).get("sha256")
            if digest and digest != _input_digest(config.get("dataset", {}).get(key))["sha256"]:
                raise ValueError(f"evaluation {key} content differs from the original training input")


def validate_inputs(config: dict[str, Any]) -> None:
    """Fail before creating a run when required local dataset paths are absent."""
    dataset = config.get("dataset", {})
    for key, is_directory in (("id_list_path", False), ("split_path", False),
                              ("interim_graph_dir", True), ("pdb_root", True)):
        value = dataset.get(key)
        if not value:
            raise ValueError(f"dataset.{key} is required")
        path = Path(value)
        if not (path.is_dir() if is_directory else path.is_file()):
            raise FileNotFoundError(f"dataset.{key} does not exist as a {'directory' if is_directory else 'file'}: {path}")


@contextlib.contextmanager
def run_lock(output: Path, *, directory: Path | None = None) -> Iterator[None]:
    # Adjacent locks also coordinate different checkouts that share an explicit
    # output path while having different machine-local runs roots.
    directory = directory if directory is not None else output.resolve().parent / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(str(output.resolve()).encode()).hexdigest()
    with (directory / f"{key}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another process is using this experiment: {output}") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def execute(config: dict[str, Any], *, run_name: str | None = None,
            dry_run: bool = False, build_cache_only: bool = False,
            eval_only: bool = False) -> dict[str, Any]:
    """Resolve and run one experiment; refuse to overwrite training artifacts."""
    if build_cache_only and eval_only:
        raise ValueError("--build-cache-only and --eval-only are mutually exclusive")
    config = prepare_config(config, run_name)
    if dry_run:
        print(json.dumps(config, indent=2, sort_keys=True))
        return config
    validate_inputs(config)
    if not build_cache_only and config.get("try_gpu", True):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable; use --set try_gpu=false for an intentional CPU run")
        gpu = int(config.get("gpu", 0))
        if gpu < 0 or gpu >= torch.cuda.device_count():
            raise ValueError(f"GPU index {gpu} is outside the visible CUDA devices")
    output = Path(config["output_dir"])
    if build_cache_only:
        # Cache preparation has no model, output directory, or GPU requirement.
        cache = Path(config.get("dataset", {}).get("cache_path") or output)
        with run_lock(cache, directory=cache.parent / ".locks"), configured_trainer(str(config["method"])):
            shared.train_model(config, build_cache_only=True)
        return config
    mode = "evaluation" if eval_only else "training"
    with run_lock(output):
        if eval_only:
            validate_evaluation(config, output)
        elif output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError(f"refusing to overwrite a nonempty experiment: {output}; choose a new --run-name")
        metadata = run_metadata(config, mode)
        output.mkdir(parents=True, exist_ok=True)
        suffix = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        metadata_path = output / (f"eval_metadata_{suffix}.json" if eval_only else "run_metadata.json")
        if not eval_only:
            shared.atomic_write_json(output / "config.resolved.json", config)
        shared.atomic_write_json(metadata_path, metadata)
        try:
            with configured_trainer(str(config["method"])):
                if config["method"] == "deepinteract" and config.get("try_gpu", True):
                    fraction = float(config.get("cuda_memory_fraction", 0.65))
                    torch.cuda.set_per_process_memory_fraction(fraction, int(config.get("gpu", 0)))
                shared.train_model(config, eval_only=eval_only)
            metadata["status"] = "completed"
        except BaseException as error:
            metadata.update(status="failed", error=f"{type(error).__name__}: {error}")
            raise
        finally:
            metadata["finished_at"] = _utc_now()
            shared.atomic_write_json(metadata_path, metadata)
    return config


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="versioned experiment YAML/JSON template")
    parser.add_argument("--run-name", help="unique directory name below this machine's runs root")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--dry-run", action="store_true", help="print resolved config without reading data or probing GPUs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--build-cache-only", action="store_true")
    mode.add_argument("--eval-only", action="store_true", help="re-evaluate this run's best checkpoint; rewrites final metrics")
    args = parser.parse_args(argv)
    config = shared.apply_overrides(shared.load_config(args.config), args.set)
    execute(config, run_name=args.run_name, dry_run=args.dry_run,
            build_cache_only=args.build_cache_only, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
