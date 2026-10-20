#!/usr/bin/env python3
"""Check this machine's storage, runtime and optional AsEP inputs without training."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines.runtime import machine_paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    paths = machine_paths()
    report = {"python": sys.executable, "paths": {k: str(v) for k, v in paths.items()}}
    errors = []
    try:
        import torch
        report.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                      cuda_available=torch.cuda.is_available())
        report["gpus"] = [dict(index=i, name=torch.cuda.get_device_name(i),
                              memory_gib=round(torch.cuda.get_device_properties(i).total_memory / 2**30, 2))
                          for i in range(torch.cuda.device_count())]
        if args.require_cuda and not torch.cuda.is_available():
            errors.append("CUDA is required but unavailable")
    except ImportError:
        errors.append("PyTorch is not installed in this Python environment")
    if args.check_data:
        data = paths["data_root"]
        required = [data / name for name in ("asepv1-AbDb-IDs.txt", "split_dict.pt",
                                             "asepv1_interim_graphs", "structures")]
        for path in required:
            if not path.exists():
                errors.append(f"Missing input: {path}")
        if not any(not p.exists() for p in required):
            id_file, split_file, graphs, structures = required
            ids = [s.strip() for s in id_file.read_text().splitlines() if s.strip()]
            missing_graphs = [i for i in ids if not (graphs / f"{i}.pt").is_file()]
            missing_pdbs = [i for i in ids if not (structures / f"{i}.pdb").is_file()]
            report["dataset"] = dict(samples=len(ids),
                                     id_list_sha256=hashlib.sha256(id_file.read_bytes()).hexdigest(),
                                     split_sha256=hashlib.sha256(split_file.read_bytes()).hexdigest(),
                                     missing_graphs=len(missing_graphs), missing_structures=len(missing_pdbs))
            if len(ids) != len(set(ids)) or not ids:
                errors.append("AsEP ID list is empty or contains duplicates")
            if missing_graphs:
                errors.append(f"Missing graphs (first 5): {missing_graphs[:5]}")
            if missing_pdbs:
                errors.append(f"Missing structures (first 5): {missing_pdbs[:5]}")
    report["errors"] = errors
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return int(bool(errors))


if __name__ == "__main__":
    raise SystemExit(main())
