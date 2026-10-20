"""Train and evaluate the isolated SE(3) geometric baseline suite."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import os.path as osp
import random
import time
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    average_precision_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import Tensor
from torch.optim import lr_scheduler
from tqdm import tqdm

from baselines.gf3drope_rotaug.augmentation import apply_random_global_rotation
from baselines.se3bench.data import load_or_build_dataset
from baselines.runtime import resolve_config


METHODS = (
    "lf3drope",
    "geformer",
    "gf3drope",
    "distance_rbf",
    "local_geometry",
    "ipa",
    "gvp",
    "gf3drope_rotaug",
)
# The project has explicitly authorized both the original 0/1/2/7 allocation
# and the later 3/4/5/6 allocation.  Individual launchers enforce the subset
# assigned to a particular sweep.
ALLOWED_GPUS = tuple(range(8))
JITTER_STANDARDS = (0.05, 0.10, 0.15, 0.20)


def feature_flags_from_bits(bits: Any) -> Dict[str, bool]:
    """Decode the PLM|aux|CDR|coordinate-jitter experiment bits."""

    value = str(bits)
    if len(value) != 4 or any(character not in "01" for character in value):
        raise ValueError(
            "bits must be a four-character PLM|aux|CDR|jitter string; "
            f"got {bits!r}"
        )
    return {
        "use_plm": value[0] == "1",
        "use_aux": value[1] == "1",
        "use_cdr_type": value[2] == "1",
        "use_coord_jitter": value[3] == "1",
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw = handle.read()
    # Generated JSON is valid YAML, but PyYAML treats JSON's 1e-05 as a
    # string.  Preserve JSON numeric types before falling back to YAML.
    try:
        config = json.loads(raw)
    except json.JSONDecodeError:
        config = yaml.safe_load(raw)
    if not isinstance(config, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return resolve_config(config)


def parse_override_value(raw: str) -> Any:
    return yaml.safe_load(raw)


def apply_overrides(config: Dict[str, Any], overrides: Sequence[str]) -> Dict[str, Any]:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"override must look like key=value, got {item!r}")
        key, raw = item.split("=", 1)
        cursor = config
        parts = key.split(".")
        for part in parts[:-1]:
            child = cursor.setdefault(part, {})
            if not isinstance(child, dict):
                raise ValueError(f"cannot assign nested override below {part!r}")
            cursor = child
        cursor[parts[-1]] = parse_override_value(raw)
    return config


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, Path):
        return str(value)
    return value


def experiment_config_fingerprint(config: Dict[str, Any]) -> str:
    """Hash the resolved scientific protocol, excluding only placement paths."""

    payload = json.loads(json.dumps(to_jsonable(config), sort_keys=True))
    payload.pop("gpu", None)
    payload.pop("output_dir", None)
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def implementation_fingerprint(method: str) -> Tuple[str, List[str]]:
    """Hash the source files that define this benchmark implementation."""

    baseline_root = Path(__file__).resolve().parents[1]
    relative_paths = [
        "se3bench/train.py",
        "se3bench/data.py",
        "se3bench/common.py",
        f"{method}/model.py",
        "gf3drope_rotaug/augmentation.py",
    ]
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        path = baseline_root / relative_path
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest(), relative_paths


def atomic_write_json(path: Path, value: Any) -> None:
    """Publish JSON only after a complete, flushed temporary write."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(to_jsonable(value), handle, indent=2, sort_keys=True)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_torch_save(path: Path, value: Any) -> None:
    """Publish a torch artifact atomically within its destination directory."""

    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def binary_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    labels = labels.astype(np.int64)
    probabilities = probabilities.astype(np.float64)
    if labels.size == 0:
        raise ValueError("cannot compute metrics for an empty target")
    output: Dict[str, float] = {}
    if 0 < labels.sum() < labels.size:
        output["auprc"] = float(average_precision_score(labels, probabilities))
        output["auroc"] = float(roc_auc_score(labels, probabilities))
    else:
        output["auprc"] = float("nan")
        output["auroc"] = float("nan")

    predictions = (probabilities >= threshold).astype(np.int64)
    output["mcc"] = (
        float(matthews_corrcoef(labels, predictions))
        if len(np.unique(labels)) > 1
        else 0.0
    )
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="binary", zero_division=0
    )
    output.update(
        {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "threshold": float(threshold),
            "tp": int(((predictions == 1) & (labels == 1)).sum()),
            "tn": int(((predictions == 0) & (labels == 0)).sum()),
            "fp": int(((predictions == 1) & (labels == 0)).sum()),
            "fn": int(((predictions == 0) & (labels == 1)).sum()),
        }
    )
    return output


def _mcc_from_counts(tp: int, tn: int, fp: int, fn: int) -> float:
    denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denominator <= 0:
        return 0.0
    return (tp * tn - fp * fn) / math.sqrt(float(denominator))


def find_best_mcc_threshold(
    probabilities: np.ndarray, labels: np.ndarray
) -> Tuple[float, float]:
    """Find the exact validation MCC optimum over all unique predictions."""

    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probabilities.shape != labels.shape or probabilities.size == 0:
        raise ValueError("probabilities and labels must be non-empty matching vectors")
    if labels.sum() == 0 or labels.sum() == labels.size:
        return 0.5, 0.0

    order = np.argsort(-probabilities, kind="stable")
    sorted_prob = probabilities[order]
    sorted_labels = labels[order]
    tp = fp = 0
    fn = int(sorted_labels.sum())
    tn = int(sorted_labels.size - fn)
    best_threshold = float(np.nextafter(sorted_prob[0], np.inf))
    best_mcc = 0.0
    index = 0
    while index < sorted_prob.size:
        end = index + 1
        while end < sorted_prob.size and sorted_prob[end] == sorted_prob[index]:
            end += 1
        positives = int(sorted_labels[index:end].sum())
        negatives = int(end - index - positives)
        tp += positives
        fn -= positives
        fp += negatives
        tn -= negatives
        score = _mcc_from_counts(tp, tn, fp, fn)
        threshold = float(sorted_prob[index])
        if score > best_mcc + 1.0e-15:
            best_mcc = score
            best_threshold = threshold
        index = end
    return best_threshold, float(best_mcc)


def format_metrics(prefix: str, loss: float, metrics: Dict[str, float]) -> str:
    return (
        f"{prefix} loss={loss:.4f} "
        f"AUPRC={metrics['auprc']:.4f} AUROC={metrics['auroc']:.4f} "
        f"MCC={metrics['mcc']:.4f} P={metrics['precision']:.4f} "
        f"R={metrics['recall']:.4f} F1={metrics['f1']:.4f} "
        f"thr={metrics['threshold']:.6f}"
    )


class SampleLoader(Iterable[Any]):
    """Minimal batch-size-one loader that keeps variable protein sizes intact."""

    def __init__(self, samples: Sequence[Any], shuffle: bool) -> None:
        self.samples = list(samples)
        self.shuffle = bool(shuffle)

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Any]:
        if self.shuffle:
            indices = torch.randperm(len(self.samples)).tolist()
        else:
            indices = range(len(self.samples))
        for index in indices:
            yield self.samples[index]


def prepare_sample(sample: Any, device: torch.device) -> Dict[str, Tensor]:
    output = {
        "x_b": sample.x_b.unsqueeze(0).to(device),
        "x_g": sample.x_g.unsqueeze(0).to(device),
        "y_g": sample.y_g.unsqueeze(0).float().to(device),
        "coords_ab": sample.coords_ca_ab.unsqueeze(0).to(device),
        "coords_ag": sample.coords_ca_ag.unsqueeze(0).to(device),
        "coords_n_ab": sample.coords_n_ab.unsqueeze(0).to(device),
        "coords_c_ab": sample.coords_c_ab.unsqueeze(0).to(device),
        "coords_n_ag": sample.coords_n_ag.unsqueeze(0).to(device),
        "coords_c_ag": sample.coords_c_ag.unsqueeze(0).to(device),
        "mask_b": sample.mask_b.unsqueeze(0).bool().to(device),
        "mask_g": sample.mask_g.unsqueeze(0).bool().to(device),
        "coord_mask_ab": sample.coord_mask_ab.unsqueeze(0).bool().to(device),
        "coord_mask_ag": sample.coord_mask_ag.unsqueeze(0).bool().to(device),
        "cdr_type": sample.cdr_type.unsqueeze(0).long().to(device),
        "edge_index_ab": sample.edge_index_ab.long().to(device),
        "edge_index_ag": sample.edge_index_ag.long().to(device),
    }
    return output


def jitter_coordinates(coords: Tensor, mask: Tensor, std: float) -> Tensor:
    if std <= 0:
        return coords
    return coords + torch.randn_like(coords) * std * mask.unsqueeze(-1).to(coords.dtype)


def apply_train_augmentation(
    batch: Dict[str, Tensor], hparams: Dict[str, Any]
) -> Dict[str, Tensor]:
    augmentation = hparams.get("augmentation", {}) or {}
    if not bool(augmentation.get("enabled", False)):
        return batch
    output = dict(batch)

    jitter_std = float(augmentation.get("coord_jitter_std", 0.0) or 0.0)
    jitter_probability = float(
        augmentation.get("coord_jitter_prob", 1.0) or 0.0
    )
    if jitter_std > 0 and random.random() < jitter_probability:
        for prefix, mask_key in (("ab", "coord_mask_ab"), ("ag", "coord_mask_ag")):
            mask = output[mask_key]
            for key in (
                f"coords_{prefix}",
                f"coords_n_{prefix}",
                f"coords_c_{prefix}",
            ):
                output[key] = jitter_coordinates(output[key], mask, jitter_std)

    rotation_probability = float(
        augmentation.get("global_rotation_prob", 0.0) or 0.0
    )
    if rotation_probability > 0:
        output = apply_random_global_rotation(
            output, probability=rotation_probability
        )

    feature_noise_std = float(
        augmentation.get("feature_noise_std", 0.0) or 0.0
    )
    if feature_noise_std > 0:
        for key, mask_key in (("x_b", "mask_b"), ("x_g", "mask_g")):
            output[key] = (
                output[key]
                + torch.randn_like(output[key])
                * feature_noise_std
                * output[mask_key].unsqueeze(-1).to(output[key].dtype)
            )

    dropout_probability = float(
        augmentation.get("feature_dropout_prob", 0.0) or 0.0
    )
    if dropout_probability > 0:
        if not 0 <= dropout_probability < 1:
            raise ValueError("feature_dropout_prob must lie in [0, 1)")
        for key in ("x_b", "x_g"):
            keep = (torch.rand_like(output[key]) >= dropout_probability).to(
                output[key].dtype
            )
            output[key] = output[key] * keep / (1.0 - dropout_probability)
    return output


def masked_bce_loss(
    logits: Tensor, target: Tensor, mask: Tensor, pos_weight: Tensor
) -> Tensor:
    loss = F.binary_cross_entropy_with_logits(
        logits, target, pos_weight=pos_weight, reduction="none"
    )
    loss = loss * mask.to(loss.dtype)
    return loss.sum() / mask.sum().clamp_min(1).to(loss.dtype)


def masked_dice_loss(
    logits: Tensor, target: Tensor, mask: Tensor, smooth: float = 1.0
) -> Tensor:
    valid = mask.to(logits.dtype)
    probability = torch.sigmoid(logits) * valid
    target = target * valid
    intersection = (probability * target).sum()
    return 1.0 - (2.0 * intersection + smooth) / (
        probability.sum() + target.sum() + smooth
    )


def masked_focal_loss(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    alpha: float,
    gamma: float,
) -> Tensor:
    probability = torch.sigmoid(logits)
    cross_entropy = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none"
    )
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * cross_entropy
    loss = loss * mask.to(loss.dtype)
    return loss.sum() / mask.sum().clamp_min(1).to(loss.dtype)


def compute_loss(
    logits: Tensor,
    target: Tensor,
    mask: Tensor,
    pos_weight: Tensor,
    hparams: Dict[str, Any],
) -> Tensor:
    loss = float(hparams.get("bce_weight", 1.0)) * masked_bce_loss(
        logits, target, mask, pos_weight
    )
    dice_weight = float(hparams.get("dice_weight", 0.0) or 0.0)
    if dice_weight:
        loss = loss + dice_weight * masked_dice_loss(
            logits,
            target,
            mask,
            smooth=float(hparams.get("dice_smooth", 1.0)),
        )
    focal_weight = float(hparams.get("focal_weight", 0.0) or 0.0)
    if focal_weight:
        loss = loss + focal_weight * masked_focal_loss(
            logits,
            target,
            mask,
            alpha=float(hparams.get("focal_alpha", 0.25)),
            gamma=float(hparams.get("focal_gamma", 2.0)),
        )
    return loss


def run_epoch(
    model: torch.nn.Module,
    loader: SampleLoader,
    device: torch.device,
    pos_weight: Tensor,
    hparams: Dict[str, Any],
    optimizer: Optional[torch.optim.Optimizer] = None,
    grad_clip: float = 1.0,
    description: str = "train",
) -> Tuple[float, Dict[str, float], np.ndarray, np.ndarray, float]:
    is_training = optimizer is not None
    model.train(is_training)
    losses: List[float] = []
    probabilities: List[Tensor] = []
    labels: List[Tensor] = []
    start = time.perf_counter()
    iterator = tqdm(
        loader,
        desc=description,
        leave=False,
        disable=not bool(hparams.get("progress", False)),
    )
    for sample in iterator:
        batch = prepare_sample(sample, device)
        if is_training:
            batch = apply_train_augmentation(batch, hparams)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_training):
            logits = model(
                x_b=batch["x_b"],
                x_g=batch["x_g"],
                coords_ab=batch["coords_ab"],
                coords_ag=batch["coords_ag"],
                coords_n_ab=batch["coords_n_ab"],
                coords_c_ab=batch["coords_c_ab"],
                coords_n_ag=batch["coords_n_ag"],
                coords_c_ag=batch["coords_c_ag"],
                cdr_type=batch["cdr_type"],
                mask_b=batch["mask_b"],
                mask_g=batch["mask_g"],
                coord_mask_ab=batch["coord_mask_ab"],
                coord_mask_ag=batch["coord_mask_ag"],
                edge_index_ab=batch["edge_index_ab"],
                edge_index_ag=batch["edge_index_ag"],
            )
            loss = compute_loss(
                logits, batch["y_g"], batch["mask_g"], pos_weight, hparams
            )
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"non-finite {description} loss for {sample.abdbid}: {loss.item()}"
                )
            if is_training:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
        losses.append(float(loss.detach().item()))
        valid = batch["mask_g"].bool()
        probabilities.append(torch.sigmoid(logits.detach())[valid].cpu())
        labels.append(batch["y_g"][valid].detach().cpu())

    elapsed = time.perf_counter() - start
    probability_array = torch.cat(probabilities).numpy()
    label_array = torch.cat(labels).numpy()
    metrics = binary_metrics(probability_array, label_array)
    mean_loss = float(np.mean(losses)) if losses else float("nan")
    return mean_loss, metrics, probability_array, label_array, elapsed


def make_loaders(
    data_list: Sequence[Any],
    split_indices: Dict[str, Tensor],
    debug_limit: int = 0,
) -> Tuple[List[Any], List[Any], List[Any], SampleLoader, SampleLoader, SampleLoader]:
    indices = {
        name: split_indices[name].tolist() for name in ("train", "val", "test")
    }
    if debug_limit > 0:
        for name in indices:
            indices[name] = indices[name][:debug_limit]
    samples = {
        name: [data_list[int(index)] for index in indices[name]]
        for name in ("train", "val", "test")
    }
    return (
        samples["train"],
        samples["val"],
        samples["test"],
        SampleLoader(samples["train"], shuffle=True),
        SampleLoader(samples["val"], shuffle=False),
        SampleLoader(samples["test"], shuffle=False),
    )


def build_model(
    method: str,
    input_ab_dim: int,
    input_ag_dim: int,
    hparams: Dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
    module = importlib.import_module(f"baselines.{method}.model")
    model_class = module.Model
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
        use_cdr_type=bool(hparams.get("use_cdr_type", True)),
        use_global_context=bool(hparams.get("use_global_context", True)),
        use_pair_mil=bool(hparams.get("use_pair_mil", True)),
        mil_weight=float(hparams.get("mil_weight", 0.75)),
        geometry_config=dict(hparams.get("geometry", {}) or {}),
    ).to(device)


def _validate_experiment_config(config: Dict[str, Any]) -> None:
    method = str(config.get("method"))
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    gpu = int(config.get("gpu", 0))
    if gpu < 0:
        raise ValueError("gpu must be a nonnegative visible CUDA device index")
    if not config.get("output_dir"):
        raise ValueError("output_dir must be a nonempty path")
    rotation_probability = float(
        config["hparams"].get("augmentation", {}).get(
            "global_rotation_prob", 0.0
        )
        or 0.0
    )
    if method == "gf3drope_rotaug" and rotation_probability <= 0:
        raise ValueError("gf3drope_rotaug requires global_rotation_prob > 0")
    if method != "gf3drope_rotaug" and rotation_probability != 0:
        raise ValueError("global rotation is only valid for gf3drope_rotaug")

    if "bits" not in config:
        return
    flags = feature_flags_from_bits(config["bits"])
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset configuration must be a mapping")
    if dataset:
        expected_ab_embedding = "igfold" if flags["use_plm"] else "none"
        expected_ag_embedding = "esm2" if flags["use_plm"] else "none"
        actual_ab_embedding = str(
            dataset.get("ab_embedding_model", expected_ab_embedding)
        ).lower()
        actual_ag_embedding = str(
            dataset.get("ag_embedding_model", expected_ag_embedding)
        ).lower()
        if actual_ab_embedding != expected_ab_embedding:
            raise ValueError(
                f"bits={config['bits']} requires antibody embedding "
                f"{expected_ab_embedding!r}, got {actual_ab_embedding!r}"
            )
        if actual_ag_embedding != expected_ag_embedding:
            raise ValueError(
                f"bits={config['bits']} requires antigen embedding "
                f"{expected_ag_embedding!r}, got {actual_ag_embedding!r}"
            )
        actual_aux = bool(
            dataset.get("include_aux_features", flags["use_aux"])
        )
        if actual_aux != flags["use_aux"]:
            raise ValueError(
                f"bits={config['bits']} requires include_aux_features="
                f"{flags['use_aux']}, got {actual_aux}"
            )

    hparams = config.get("hparams", {})
    if not isinstance(hparams, dict):
        raise ValueError("hparams configuration must be a mapping")
    actual_cdr = bool(hparams.get("use_cdr_type", flags["use_cdr_type"]))
    if actual_cdr != flags["use_cdr_type"]:
        raise ValueError(
            f"bits={config['bits']} requires use_cdr_type="
            f"{flags['use_cdr_type']}, got {actual_cdr}"
        )

    augmentation = hparams.get("augmentation", {}) or {}
    if not isinstance(augmentation, dict):
        raise ValueError("hparams.augmentation must be a mapping")
    enabled = bool(
        augmentation.get("enabled", flags["use_coord_jitter"])
    )
    jitter_std = float(
        augmentation.get(
            "coord_jitter_std", 0.20 if flags["use_coord_jitter"] else 0.0
        )
        or 0.0
    )
    jitter_probability = float(
        augmentation.get(
            "coord_jitter_prob", 1.0 if flags["use_coord_jitter"] else 0.0
        )
        or 0.0
    )
    if flags["use_coord_jitter"]:
        valid_std = any(
            math.isclose(jitter_std, expected, rel_tol=0.0, abs_tol=1.0e-12)
            for expected in JITTER_STANDARDS
        )
        if not enabled or not valid_std or not math.isclose(
            jitter_probability, 1.0, rel_tol=0.0, abs_tol=1.0e-12
        ):
            raise ValueError(
                f"bits={config['bits']} requires enabled coordinate jitter with "
                f"std in {JITTER_STANDARDS} and probability 1.0"
            )
    elif enabled or not math.isclose(
        jitter_std, 0.0, rel_tol=0.0, abs_tol=1.0e-12
    ) or not math.isclose(
        jitter_probability, 0.0, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError(
            f"bits={config['bits']} requires coordinate jitter to be disabled "
            "with std=0 and probability=0"
        )


def train_model(
    config: Dict[str, Any],
    *,
    eval_only: bool = False,
    build_cache_only: bool = False,
) -> None:
    config = resolve_config(config)
    _validate_experiment_config(config)
    print("Config:\n" + pformat(config), flush=True)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    if config.get("num_threads"):
        torch.set_num_threads(int(config["num_threads"]))

    use_cuda = torch.cuda.is_available() and bool(config.get("try_gpu", True))
    gpu = int(config.get("gpu", 0))
    device = torch.device(f"cuda:{gpu}" if use_cuda else "cpu")
    print(f"Using device: {device}", flush=True)

    dataset_config = dict(config["dataset"])
    data_list, split_indices, metadata = load_or_build_dataset(dataset_config)
    if build_cache_only:
        print(
            f"Cache ready: samples={len(data_list)} metadata={pformat(metadata)}",
            flush=True,
        )
        return

    hparams = config["hparams"]
    debug_limit = int(config.get("debug_limit", 0) or 0)
    (
        train_samples,
        val_samples,
        test_samples,
        train_loader,
        val_loader,
        test_loader,
    ) = make_loaders(data_list, split_indices, debug_limit)
    print(
        f"Dataset total={len(data_list)} train={len(train_samples)} "
        f"val={len(val_samples)} test={len(test_samples)}",
        flush=True,
    )
    print("Dataset metadata:\n" + pformat(metadata), flush=True)

    positives = int(sum(sample.y_g.sum().item() for sample in train_samples))
    total = int(sum(sample.y_g.numel() for sample in train_samples))
    negatives = total - positives
    automatic_weight = negatives / max(positives, 1)
    configured_weight = hparams.get("pos_weight")
    weight_value = (
        automatic_weight if configured_weight is None else float(configured_weight)
    )
    pos_weight = torch.tensor(weight_value, dtype=torch.float32, device=device)
    print(
        f"Train labels pos={positives} neg={negatives} "
        f"pos_weight={weight_value:.6f}",
        flush=True,
    )

    sample = train_samples[0]
    method = str(config["method"])
    model = build_model(
        method, sample.x_b.size(-1), sample.x_g.size(-1), hparams, device
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model parameters: {parameter_count:,}", flush=True)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "best_model.pt"
    history_path = output_dir / "history.jsonl"

    optimizer_config = config.get("optimizer", {})
    optimizer_class = getattr(
        torch.optim, optimizer_config.get("name", "AdamW")
    )
    optimizer = optimizer_class(
        model.parameters(),
        **optimizer_config.get(
            "params", {"lr": 1.0e-4, "weight_decay": 1.0e-5}
        ),
    )
    scheduler = None
    scheduler_config = config.get("callbacks", {}).get("lr_scheduler")
    if scheduler_config:
        scheduler_class = getattr(lr_scheduler, scheduler_config["name"])
        scheduler = scheduler_class(
            optimizer, **scheduler_config.get("kwargs", {})
        )

    max_epochs = int(hparams.get("max_epochs", 100))
    patience = int(
        config.get("callbacks", {})
        .get("early_stopping", {})
        .get("patience", 15)
    )
    monitor = str(config.get("monitor", "auprc"))
    gradient_clip = float(hparams.get("grad_clip", 1.0))
    best_epoch = -1

    if eval_only:
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
    else:
        best_score = -float("inf")
        patience_counter = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with history_path.open("w", encoding="utf-8") as history:
            for epoch in range(1, max_epochs + 1):
                print(f"\n=== Epoch {epoch}/{max_epochs} ===", flush=True)
                (
                    train_loss,
                    train_metrics,
                    _,
                    _,
                    train_seconds,
                ) = run_epoch(
                    model,
                    train_loader,
                    device,
                    pos_weight,
                    hparams,
                    optimizer=optimizer,
                    grad_clip=gradient_clip,
                    description="train",
                )
                val_loss, val_metrics, _, _, val_seconds = run_epoch(
                    model,
                    val_loader,
                    device,
                    pos_weight,
                    hparams,
                    description="val",
                )
                print(format_metrics("Train", train_loss, train_metrics), flush=True)
                print(format_metrics("Val  ", val_loss, val_metrics), flush=True)
                score = float(val_metrics[monitor])
                if not math.isfinite(score):
                    raise FloatingPointError(
                        f"validation monitor {monitor} is non-finite at epoch {epoch}"
                    )
                if scheduler is not None:
                    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(score)
                    else:
                        scheduler.step()

                row = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "train_seconds": train_seconds,
                    "val_seconds": val_seconds,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                history.write(
                    json.dumps(to_jsonable(row), sort_keys=True) + "\n"
                )
                history.flush()

                if score > best_score:
                    best_score = score
                    best_epoch = epoch
                    patience_counter = 0
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "epoch": epoch,
                            "best_score": best_score,
                            "monitor": monitor,
                            "config": config,
                            "dataset_metadata": metadata,
                            "input_ab_dim": int(sample.x_b.size(-1)),
                            "input_ag_dim": int(sample.x_g.size(-1)),
                            "parameter_count": parameter_count,
                        },
                        checkpoint_path,
                    )
                    print(
                        f"  -> saved {checkpoint_path} "
                        f"({monitor}={best_score:.6f})",
                        flush=True,
                    )
                else:
                    patience_counter += 1
                if patience_counter >= patience:
                    print(
                        f"Early stopping at epoch {epoch}; "
                        f"best_epoch={best_epoch}, best_{monitor}={best_score:.6f}",
                        flush=True,
                    )
                    break

    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    val_loss, val_metrics_05, val_probs, val_labels, val_seconds = run_epoch(
        model, val_loader, device, pos_weight, hparams, description="val-best"
    )
    best_threshold, best_val_mcc = find_best_mcc_threshold(
        val_probs, val_labels
    )
    val_metrics_tuned = binary_metrics(
        val_probs, val_labels, threshold=best_threshold
    )
    (
        test_loss,
        test_metrics_05,
        test_probs,
        test_labels,
        test_seconds,
    ) = run_epoch(
        model, test_loader, device, pos_weight, hparams, description="test"
    )
    test_metrics_tuned = binary_metrics(
        test_probs, test_labels, threshold=best_threshold
    )
    print(format_metrics("Val@0.5  ", val_loss, val_metrics_05), flush=True)
    print(format_metrics("Val@best ", val_loss, val_metrics_tuned), flush=True)
    print(format_metrics("Test@0.5 ", test_loss, test_metrics_05), flush=True)
    print(format_metrics("Test@best", test_loss, test_metrics_tuned), flush=True)

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    peak_reserved_memory = (
        int(torch.cuda.max_memory_reserved(device))
        if device.type == "cuda"
        else 0
    )
    code_fingerprint, implementation_files = implementation_fingerprint(method)
    result = {
        "method": method,
        "split": config.get("split"),
        "bits": str(config.get("bits")),
        "seed": seed,
        "best_epoch": int(checkpoint["epoch"]),
        "best_score": float(checkpoint["best_score"]),
        "monitor": checkpoint["monitor"],
        "best_threshold_by_val_mcc": float(best_threshold),
        "best_val_mcc": float(best_val_mcc),
        "val_metrics_05": val_metrics_05,
        "val_metrics_tuned": val_metrics_tuned,
        "test_metrics_05": test_metrics_05,
        "test_metrics_tuned": test_metrics_tuned,
        "val_eval_seconds": val_seconds,
        "test_eval_seconds": test_seconds,
        "parameter_count": parameter_count,
        "peak_gpu_memory_bytes": peak_memory,
        "peak_gpu_memory_reserved_bytes": peak_reserved_memory,
        "dataset_metadata": metadata,
        "provenance": {
            "experiment_config_fingerprint": experiment_config_fingerprint(config),
            "implementation_fingerprint": code_fingerprint,
            "implementation_files": implementation_files,
        },
        "environment": {
            "python": os.sys.version,
            "python_executable": os.sys.executable,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": str(device),
            "pytorch_cuda_alloc_conf": os.environ.get(
                "PYTORCH_CUDA_ALLOC_CONF", ""
            ),
        },
        "config": config,
    }
    atomic_torch_save(output_dir / "final_metrics.pt", result)
    atomic_write_json(output_dir / "final_metrics.json", result)
    print(f"Saved metrics to {output_dir / 'final_metrics.json'}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config.yaml")),
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        help="Override a nested key, e.g. --set hparams.max_epochs=1",
    )
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--build-cache-only", action="store_true")
    arguments = parser.parse_args()
    configuration = apply_overrides(
        load_config(arguments.config), arguments.set
    )
    train_model(
        configuration,
        eval_only=arguments.eval_only,
        build_cache_only=arguments.build_cache_only,
    )


if __name__ == "__main__":
    main()
