"""Check portable entrypoints without GPUs or the AsEP dataset."""
import json
from types import SimpleNamespace

import pytest
import torch

from baselines import run

_validate_inputs = run.validate_inputs


@pytest.fixture
def paths(tmp_path, monkeypatch):
    roots = {key: tmp_path / key for key in ("data_root", "cache_root", "runs_root")}
    monkeypatch.setattr(run.runtime, "machine_paths", lambda: roots)
    # Most tests isolate scheduling from data IO; the CPU integration uses a
    # synthetic dataset loader. The real path boundary is exercised below.
    monkeypatch.setattr(run, "validate_inputs", lambda config: None)
    return roots


def config(method="lf3drope"):
    return {
        "method": method, "gpu": 0, "try_gpu": False, "bits": "1100",
        "dataset": {"ab_embedding_model": "igfold", "ag_embedding_model": "esm2", "include_aux_features": True},
        "hparams": {
            "use_cdr_type": False, "d_model": 32, "n_heads": 4, "n_layers_ab": 1,
            "n_layers_ag": 1, "n_layers_cross": 1, "dim_ff": 64,
            "geometry": {"n_rope_freqs": 1},
            "augmentation": {"enabled": False, "coord_jitter_std": 0., "coord_jitter_prob": 0.,
                             "global_rotation_prob": 1. if method == "gf3drope_rotaug" else 0.},
        },
    }


def test_dry_run_has_no_data_gpu_or_filesystem_effects(paths, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        pytest.fail("dry run accessed data or GPU")
    monkeypatch.setattr(torch.cuda, "is_available", forbidden)
    monkeypatch.setattr(run.shared, "train_model", forbidden)
    original = config()
    original["try_gpu"] = True
    resolved = run.execute(original, run_name="trial", dry_run=True)
    assert resolved["output_dir"] == str(paths["runs_root"] / "trial")
    assert "output_dir" not in original
    assert json.loads(capsys.readouterr().out) == resolved
    assert not paths["runs_root"].exists()


@pytest.mark.parametrize("name", ["../escape", "/absolute", "a/b", ".", "..", "bad name"])
def test_run_names_cannot_escape_machine_root(paths, name):
    with pytest.raises(ValueError, match="run-name"):
        run.prepare_config(config(), name)


def test_missing_name_and_inconsistent_features_rejected(paths):
    with pytest.raises(ValueError, match="run-name"):
        run.prepare_config(config())
    invalid = config()
    invalid["hparams"]["use_cdr_type"] = True
    with pytest.raises(ValueError):
        run.prepare_config(invalid, "invalid")


def test_split_label_must_match_actual_dataset_split(paths):
    selected = config()
    selected.update(split="ratio")
    selected["dataset"]["split_method"] = "epitope_group"
    with pytest.raises(ValueError, match="override both"):
        run.prepare_config(selected, "bad-split")


def test_missing_dataset_fails_before_creating_output(paths, monkeypatch):
    monkeypatch.setattr(run, "validate_inputs", _validate_inputs)
    selected = config()
    selected["dataset"]["id_list_path"] = str(paths["data_root"] / "missing-ids.txt")
    with pytest.raises(FileNotFoundError, match="id_list_path"):
        run.execute(selected, run_name="no-data")
    assert not paths["runs_root"].exists()


@pytest.mark.parametrize("method", run.METHODS)
def test_all_models_construct_and_hooks_restore(paths, method):
    selected = config(method)
    if method == "1drope":
        selected["hparams"]["geometry"] = {"rope_base": 10000.}
    snapshot = {name: getattr(run.shared, name) for name in run._HOOKS}
    run.prepare_config(selected, method)
    with run.configured_trainer(method):
        model = run.shared.build_model(method, 8, 7, selected["hparams"], torch.device("cpu"))
        assert isinstance(model, torch.nn.Module)
        if method == "deepinteract":
            assert run.shared.run_epoch.__module__ == "baselines.deepinteract.train"
    for name, expected in snapshot.items():
        assert getattr(run.shared, name) is expected


@pytest.mark.parametrize("cross", [False, True])
@pytest.mark.parametrize("global_context", [False, True])
@pytest.mark.parametrize("pair_mil", [False, True])
def test_ablation_launcher_preserves_branch_switches(paths, cross, global_context, pair_mil):
    selected = config("lf3drope_module_ablation")
    selected["hparams"].update(use_cross_attention=cross, use_global_context=global_context,
                               use_pair_mil=pair_mil)
    with run.configured_trainer(selected["method"]):
        model = run.shared.build_model(selected["method"], 8, 7, selected["hparams"], torch.device("cpu"))
    assert len(model.cross_layers) == int(cross)
    assert hasattr(model, "global_proj") is global_context
    assert hasattr(model, "pair_head") is pair_mil


def test_one_d_rope_rejects_disabled_cross_attention(paths):
    selected = config("1drope")
    selected["hparams"].update(use_cross_attention=False, geometry={"rope_base": 10000.})
    with pytest.raises(ValueError, match="use_cross_attention"):
        run.prepare_config(selected, "invalid-1drope")
    with run.configured_trainer("1drope"), pytest.raises(ValueError, match="cross-attention"):
        run.shared.build_model("1drope", 8, 7, selected["hparams"], torch.device("cpu"))


def test_failed_training_records_provenance_and_restores_hooks(paths, monkeypatch):
    snapshot = {name: getattr(run.shared, name) for name in run._HOOKS}
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic failure")
    monkeypatch.setattr(run.shared, "train_model", fail)
    ids = paths["data_root"] / "ids.txt"
    ids.parent.mkdir()
    ids.write_text("sample1\n")
    selected = config("deepinteract")
    selected["dataset"]["id_list_path"] = str(ids)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        run.execute(selected, run_name="failed")
    metadata = json.loads((paths["runs_root"] / "failed/run_metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["input_files"]["id_list_path"]["sha256"]
    assert "git" in metadata and "host" in metadata and "finished_at" in metadata
    for name, expected in snapshot.items():
        assert getattr(run.shared, name) is expected


def test_existing_output_is_never_overwritten(paths, monkeypatch):
    output = paths["runs_root"] / "existing"
    output.mkdir(parents=True)
    artifact = output / "best_model.pt"
    artifact.write_bytes(b"original")
    with pytest.raises(FileExistsError, match="overwrite"):
        run.execute(config(), run_name="existing")
    assert artifact.read_bytes() == b"original"
    assert not (output / "run_metadata.json").exists()


def test_training_lock_rejects_second_writer(paths):
    output = paths["runs_root"] / "locked"
    with run.run_lock(output):
        with pytest.raises(RuntimeError, match="another process"):
            with run.run_lock(output):
                pytest.fail("lock was not exclusive")


def test_cuda_requested_never_falls_back_silently(paths, monkeypatch):
    selected = config()
    selected["try_gpu"] = True
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA requested"):
        run.execute(selected, run_name="cuda")
    assert not paths["runs_root"].exists()


def test_cache_build_does_not_create_training_output(paths, monkeypatch):
    calls = []
    monkeypatch.setattr(run.shared, "train_model", lambda config, **kwargs: calls.append(kwargs))
    run.execute(config(), run_name="cache", build_cache_only=True)
    assert calls == [{"build_cache_only": True}]
    assert not (paths["runs_root"] / "cache").exists()


def test_evaluation_preserves_training_metadata(paths, monkeypatch):
    output = paths["runs_root"] / "evaluate"
    output.mkdir(parents=True)
    torch.save({"config": run.prepare_config(config(), "evaluate")}, output / "best_model.pt")
    (output / "run_metadata.json").write_text('{"original": true}')
    calls = []
    monkeypatch.setattr(run.shared, "train_model", lambda config, **kwargs: calls.append(kwargs))
    run.execute(config(), run_name="evaluate", eval_only=True)
    assert calls == [{"eval_only": True}]
    assert json.loads((output / "run_metadata.json").read_text()) == {"original": True}
    metadata_paths = list(output.glob("eval_metadata_*.json"))
    assert len(metadata_paths) == 1
    assert json.loads(metadata_paths[0].read_text())["status"] == "completed"


def test_evaluation_rejects_relabeling_a_checkpoint(paths, monkeypatch):
    output = paths["runs_root"] / "original"
    output.mkdir(parents=True)
    torch.save({"config": run.prepare_config(config(), "original")}, output / "best_model.pt")
    selected = config()
    selected["seed"] = 999
    with pytest.raises(ValueError, match="checkpoint science"):
        run.execute(selected, run_name="original", eval_only=True)
    assert not list(output.glob("eval_metadata_*.json"))


def test_cpu_training_and_evaluation_roundtrip(paths, monkeypatch):
    from baselines.tests.test_models import _batch
    batch = _batch()
    sample = SimpleNamespace(
        **{key: value.squeeze(0) for key, value in batch.items()
           if key not in ("edge_index_ab", "edge_index_ag", "coords_ab", "coords_ag")},
        coords_ca_ab=batch["coords_ab"].squeeze(0),
        coords_ca_ag=batch["coords_ag"].squeeze(0),
        edge_index_ab=batch["edge_index_ab"], edge_index_ag=batch["edge_index_ag"],
        y_g=torch.tensor([0., 1., 0., 1., 0., 1.]), abdbid="synthetic",
    )
    splits = {"train": torch.tensor([0]), "val": torch.tensor([1]), "test": torch.tensor([2])}
    monkeypatch.setattr(run.shared, "load_or_build_dataset", lambda config: ([sample] * 3, splits, {"synthetic": True}))
    selected = config()
    selected.update(seed=42, num_threads=1)
    selected["hparams"].update(max_epochs=1, dropout=0.)
    run.execute(selected, run_name="smoke")
    output = paths["runs_root"] / "smoke"
    trained = json.loads((output / "final_metrics.json").read_text())
    assert trained["best_epoch"] == 1
    assert (output / "best_model.pt").is_file()
    snapshot = run.shared.load_config(str(output / "config.resolved.json"))
    run.execute(snapshot, eval_only=True)
    evaluated = json.loads((output / "final_metrics.json").read_text())
    assert trained["test_metrics_tuned"] == evaluated["test_metrics_tuned"]
