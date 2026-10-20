#!/usr/bin/env python3
"""Check upload candidates without reading or printing their contents.

Checks the Git index (including staged blob sizes) and unignored working files.
Before ``git init``, uses disposable Git metadata outside the project so Git's
actual ignore rules still apply. Does not stage, commit, or change project files.
This checks the current tree/index, not old commits or arbitrary secret content.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import math
import os
from pathlib import Path, PurePosixPath
import subprocess
import sys
import tempfile


ARTIFACT_SUFFIXES = frozenset(
    ".pt .pth .ckpt .safetensors .onnx .bin .pkl .pickle .npy .npz "
    ".h5 .hdf5 .hdf .parquet .arrow .feather .csv .tsv .jsonl .lmdb .mdb "
    ".sqlite .sqlite3 .db .pdb .cif .mmcif .fasta .fa .fastq "
    ".gz .bz2 .xz .zip .tar .tgz .7z .rar .log .out .err .pid".split()
)
ARTIFACT_DIRS = frozenset(
    ".local data dataset datasets cache caches .cache _shared checkpoints checkpoint "
    "weights outputs results remote_results predictions inference runs logs wandb lightning_logs "
    "tensorboard __pycache__ .venv venv env .conda .pytest_cache .mypy_cache "
    ".ruff_cache .ipynb_checkpoints".split()
)
RUNTIME_METADATA_NAMES = frozenset(
    "config.resolved.json run_metadata.json eval_metadata.json final_metrics.json".split()
)


def run_git(command: list[str], *args: str, data: bytes | None = None,
            allowed: tuple[int, ...] = (0,)) -> bytes:
    result = subprocess.run(command + list(args), input=data, capture_output=True)
    if result.returncode not in allowed:
        # Git stderr can contain arbitrary paths; keep failures concise.
        raise RuntimeError(f"Git {' '.join(args[:1])} failed (exit {result.returncode})")
    return result.stdout


@contextmanager
def git_context(root: Path):
    command = ["git", "-C", str(root)]
    probe = subprocess.run(command + ["rev-parse", "--show-toplevel"],
                           capture_output=True)
    if probe.returncode == 0 and Path(os.fsdecode(probe.stdout).strip()).resolve() == root:
        yield command, True
        return
    with tempfile.TemporaryDirectory(prefix="epitope-git-check-") as metadata:
        run_git(["git"], "init", "--bare", "--quiet", metadata)
        yield ["git", "--git-dir", metadata, "--work-tree", str(root), "-C", str(root)], False


def forbidden_path(name: str) -> str | None:
    path = PurePosixPath(name)
    lowered = path.name.lower()
    dirs = [part.lower() for part in path.parts[:-1]]
    if any(part in ARTIFACT_DIRS or part.startswith(
        ("outputs_", "results_", "predictions_", "inference_")
    ) for part in dirs):
        return "local data, environment, cache, or output directory"
    if lowered in RUNTIME_METADATA_NAMES or (
        lowered.startswith("eval_metadata_") and lowered.endswith(".json")
    ):
        return "generated runtime configuration or metrics"
    if path.suffix.lower() in ARTIFACT_SUFFIXES:
        return "dataset, checkpoint, archive, or generated-output suffix"
    if lowered.startswith(".env") and lowered != ".env.example":
        return "local environment/secret file"
    if lowered == "machine.local.json" or lowered.startswith("config.local.") or lowered.endswith(
        (".local.json", ".local.yaml", ".local.yml")
    ):
        return "machine-specific configuration"
    return None


def inspect(root: Path, max_bytes: int) -> tuple[list[tuple[str, str]], int, bool]:
    findings: set[tuple[str, str]] = set()
    with git_context(root) as (command, has_index):
        entries = run_git(command, "ls-files", "--cached", "--stage", "-z").split(b"\0")
        indexed: dict[str, list[str]] = {}
        for entry in entries:
            if not entry:
                continue
            metadata, raw_name = entry.split(b"\t", 1)
            mode, object_id, _stage = metadata.split()
            name = os.fsdecode(raw_name)
            indexed.setdefault(name, [])
            if mode != b"160000":  # A submodule entry is a commit, not a file blob.
                indexed[name].append(object_id.decode("ascii"))
        untracked = run_git(command, "ls-files", "--others", "--exclude-standard", "-z")
        candidates = set(indexed) | {os.fsdecode(item) for item in untracked.split(b"\0") if item}
        if candidates:
            ignored_output = run_git(
                command, "check-ignore", "--no-index", "--stdin", "-z",
                data=b"\0".join(os.fsencode(item) for item in sorted(candidates)) + b"\0",
                allowed=(0, 1),
            )
            ignored = {os.fsdecode(item) for item in ignored_output.split(b"\0") if item}
        else:
            ignored = set()
        object_ids = sorted({item for values in indexed.values() for item in values})
        sizes: dict[str, int] = {}
        if object_ids:
            objects = run_git(command, "cat-file", "--batch-check=%(objectname) %(objectsize)",
                              data=("\n".join(object_ids) + "\n").encode("ascii"))
            for line in objects.splitlines():
                object_id, size = line.split()
                sizes[object_id.decode("ascii")] = int(size)
        for name in sorted(candidates):
            reason = forbidden_path(name)
            if reason:
                findings.add((name, reason))
            elif name in ignored:
                findings.add((name, "already tracked/staged despite ignore rules"))
            path = root / name
            if path.is_symlink():
                findings.add((name, "symbolic link; commit source files, keep local data links ignored"))
            elif path.is_file():
                size = path.stat().st_size
                if size > max_bytes:
                    findings.add((name, f"working file exceeds limit ({size:,} bytes)"))
            if any(sizes[object_id] > max_bytes for object_id in indexed.get(name, [])):
                findings.add((name, "staged/tracked blob exceeds size limit"))
    return sorted(findings), len(candidates), has_index


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--max-size-mb", type=float, default=10.0,
                        help="maximum candidate or index-blob size in MiB (default: 10)")
    args = parser.parse_args(argv)
    if not math.isfinite(args.max_size_mb) or args.max_size_mb <= 0:
        parser.error("--max-size-mb must be a positive finite number")
    root = args.root.resolve()
    if not root.is_dir():
        parser.error("--root must be an existing directory")
    try:
        findings, count, has_index = inspect(root, int(args.max_size_mb * 1024 * 1024))
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Repository check failed: {type(error).__name__}. Ensure Git is installed and files are readable.",
              file=sys.stderr)
        return 2
    if findings:
        for name, reason in findings:
            print(f"FAIL {json.dumps(name, ensure_ascii=True)}: {reason}")
        print(f"Rejected {len(findings)} finding(s) across {count} upload candidates.")
        print("Keep artifacts local; remove accidentally staged/tracked artifacts from the Git index.")
        return 1
    scope = "Git index + unignored files" if has_index else "unignored files; Git not initialized here"
    print(f"PASS: {count} upload candidates checked ({scope}); limit {args.max_size_mb:g} MiB.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
