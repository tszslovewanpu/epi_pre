from pathlib import Path

import pytest

from baselines.se3bench.train import (
    METHODS,
    _validate_experiment_config,
    atomic_write_json,
    experiment_config_fingerprint,
    feature_flags_from_bits,
    implementation_fingerprint,
)


def _config(method: str, rotation_probability: float):
    return {
        "method": method,
        "gpu": 0,
        "output_dir": str(
            Path(__file__).resolve().parents[1] / "_shared" / "test-output"
        ),
        "hparams": {
            "augmentation": {
                "global_rotation_prob": rotation_probability,
            }
        },
    }


def test_original_methods_are_registered():
    assert "lf3drope" in METHODS
    assert "gf3drope" in METHODS


@pytest.mark.parametrize("method", ("lf3drope", "gf3drope"))
def test_original_methods_require_unaugmented_training(method):
    _validate_experiment_config(_config(method, 0.0))
    with pytest.raises(ValueError, match="global rotation"):
        _validate_experiment_config(_config(method, 1.0))


def test_rotation_control_requires_augmentation():
    _validate_experiment_config(_config("gf3drope_rotaug", 1.0))
    with pytest.raises(ValueError, match="requires"):
        _validate_experiment_config(_config("gf3drope_rotaug", 0.0))


def test_feature_bits_preserve_leading_zeroes_and_decode_in_order():
    assert feature_flags_from_bits("0000") == {
        "use_plm": False,
        "use_aux": False,
        "use_cdr_type": False,
        "use_coord_jitter": False,
    }
    assert feature_flags_from_bits("1011") == {
        "use_plm": True,
        "use_aux": False,
        "use_cdr_type": True,
        "use_coord_jitter": True,
    }
    with pytest.raises(ValueError, match="four-character"):
        feature_flags_from_bits(1)


def test_feature_bits_reject_mismatched_dataset_and_augmentation_config():
    config = _config("lf3drope", 0.0)
    config.update(
        {
            "bits": "0000",
            "dataset": {
                "ab_embedding_model": "none",
                "ag_embedding_model": "none",
                "include_aux_features": False,
            },
        }
    )
    config["hparams"].update(
        {
            "use_cdr_type": False,
            "augmentation": {
                "enabled": False,
                "coord_jitter_std": 0.0,
                "coord_jitter_prob": 0.0,
                "global_rotation_prob": 0.0,
            },
        }
    )
    _validate_experiment_config(config)

    config["dataset"]["include_aux_features"] = True
    with pytest.raises(ValueError, match="include_aux_features"):
        _validate_experiment_config(config)

    config["dataset"]["include_aux_features"] = False
    config["hparams"]["augmentation"]["enabled"] = True
    with pytest.raises(ValueError, match="coordinate jitter to be disabled"):
        _validate_experiment_config(config)


def test_provenance_fingerprints_and_atomic_json(tmp_path):
    first = {
        "method": "lf3drope",
        "gpu": 3,
        "output_dir": "/tmp/first",
        "hparams": {"max_epochs": 100},
    }
    second = {
        **first,
        "gpu": 6,
        "output_dir": "/tmp/second",
    }
    assert experiment_config_fingerprint(first) == experiment_config_fingerprint(
        second
    )
    second["hparams"] = {"max_epochs": 99}
    assert experiment_config_fingerprint(first) != experiment_config_fingerprint(
        second
    )

    code_fingerprint, files = implementation_fingerprint("lf3drope")
    assert len(code_fingerprint) == 64
    assert "lf3drope/model.py" in files

    destination = tmp_path / "result.json"
    atomic_write_json(destination, {"complete": True})
    assert destination.read_text(encoding="utf-8").strip().startswith("{")
    assert not list(tmp_path.glob(".*.tmp"))
