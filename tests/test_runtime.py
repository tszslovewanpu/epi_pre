import json
from pathlib import Path

import pytest

from baselines import runtime


def test_machine_paths_precedence_and_cwd_independence(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    repo.mkdir()
    monkeypatch.setattr(runtime, "REPO_ROOT", repo)
    for name in ("DATA", "CACHE", "RUNS"):
        monkeypatch.delenv(f"EPITOPE_{name}_ROOT", raising=False)
    (repo / "machine.local.json").write_text(json.dumps({"data_root": "private-data"}))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EPITOPE_CACHE_ROOT", str(tmp_path / "shared-cache"))
    paths = runtime.machine_paths()
    assert paths == {"data_root": repo / "private-data", "cache_root": tmp_path / "shared-cache",
                     "runs_root": repo / ".local/runs"}
    original = {"dataset": {"split_path": "${EPITOPE_DATA_ROOT}/split_dict.pt"},
                "output_dir": "${EPITOPE_RUNS_ROOT}/trial"}
    result = runtime.resolve_config(original)
    assert result["dataset"]["split_path"] == str(repo / "private-data/split_dict.pt")
    assert result["output_dir"] == str(repo / ".local/runs/trial")
    assert original["output_dir"].startswith("${")


def test_bad_machine_key_and_unresolved_variable_fail(tmp_path, monkeypatch):
    monkeypatch.setattr(runtime, "REPO_ROOT", tmp_path)
    local = tmp_path / "machine.local.json"
    local.write_text('{"data_rooot": "data"}')
    with pytest.raises(ValueError, match="must map only"):
        runtime.machine_paths()
    local.unlink()
    monkeypatch.delenv("THIS_VARIABLE_DOES_NOT_EXIST", raising=False)
    with pytest.raises(ValueError, match="undefined variable"):
        runtime.resolve_config({"output_dir": "${THIS_VARIABLE_DOES_NOT_EXIST}/run"})
