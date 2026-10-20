"""DeepInteract full-context data and epoch hooks for ``baselines.run``."""
from __future__ import annotations

import random
import time
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.se3bench import train as shared
from baselines.runtime import machine_paths
from .data import batch_context, jitter_context, load_or_build_context_cache, selected_coordinates

_BASE_LOAD = shared.load_or_build_dataset
_BASE_PREPARE = shared.prepare_sample


def load_dataset(config: dict[str, Any]):
    samples, splits, metadata = _BASE_LOAD(config)
    cache = Path(config.get("deepinteract_context_cache", machine_paths()["cache_root"] / "asep_deepinteract_full_context_v1.pt"))
    contexts, context_metadata = load_or_build_context_cache(samples, config, cache)
    extended = [SimpleNamespace(**vars(sample), context_ab=contexts[sample.abdbid]["ab"],
                                context_ag=contexts[sample.abdbid]["ag"]) for sample in samples]
    return extended, splits, {**metadata, "deepinteract_context": context_metadata}


def sync_selected_coordinates(batch: dict[str, Any]) -> dict[str, Any]:
    for part in ("ab", "ag"):
        selected = selected_coordinates(batch[f"context_{part}"])
        batch[f"coords_n_{part}"] = selected["coords_n"]
        batch[f"coords_{part}"] = selected["coords_ca"]
        batch[f"coords_c_{part}"] = selected["coords_c"]
        batch[f"coord_mask_{part}"] = selected["coord_mask"]
    return batch


def prepare_sample(sample: Any, device: torch.device) -> dict[str, Any]:
    batch = _BASE_PREPARE(sample, device)
    batch["context_ab"] = batch_context([sample.context_ab], device=device)
    batch["context_ag"] = batch_context([sample.context_ag], device=device)
    return sync_selected_coordinates(batch)


def apply_train_augmentation(batch: dict[str, Any], hparams: dict[str, Any]) -> dict[str, Any]:
    aug = hparams.get("augmentation", {}) or {}
    if not aug.get("enabled", False):
        return batch
    if any(float(aug.get(key, 0) or 0) != 0 for key in
           ("global_rotation_prob", "feature_dropout_prob", "feature_noise_std")):
        raise ValueError("This DeepInteract protocol supports backbone jitter only")
    std = float(aug.get("coord_jitter_std", 0) or 0)
    probability = float(aug.get("coord_jitter_prob", 1) or 0)
    if std <= 0 or random.random() >= probability:
        return batch
    output = dict(batch)
    for part in ("ab", "ag"):
        output[f"context_{part}"] = jitter_context(batch[f"context_{part}"], std)
    return sync_selected_coordinates(output)


def run_epoch(model, loader, device, pos_weight, hparams, optimizer=None, grad_clip=1.0, description="train"):
    """Shared epoch semantics, with full-context arguments and finite gradients."""
    training = optimizer is not None
    model.train(training)
    losses, probabilities, labels = [], [], []
    started = time.perf_counter()
    iterator = shared.tqdm(loader, desc=description, leave=False, disable=not hparams.get("progress", False))
    for sample in iterator:
        batch = prepare_sample(sample, device)
        if training:
            batch = apply_train_augmentation(batch, hparams)
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(**{key: value for key, value in batch.items() if key != "y_g"})
            loss = shared.compute_loss(logits, batch["y_g"], batch["mask_g"], pos_weight, hparams)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"non-finite {description} loss for {sample.abdbid}: {loss.item()}")
            if training:
                loss.backward()
                # Finite checks stop a failed run before corrupting optimizer
                # state; no samples or updates are silently skipped.
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip if grad_clip > 0 else float("inf"),
                                               error_if_nonfinite=True)
                optimizer.step()
        losses.append(float(loss.detach().item()))
        mask = batch["mask_g"].bool()
        probabilities.append(torch.sigmoid(logits.detach())[mask].cpu())
        labels.append(batch["y_g"][mask].detach().cpu())
    probability_array = torch.cat(probabilities).numpy()
    label_array = torch.cat(labels).numpy()
    return (float(np.mean(losses)), shared.binary_metrics(probability_array, label_array),
            probability_array, label_array, time.perf_counter() - started)


def configure_trainer() -> None:
    if "deepinteract" not in shared.METHODS:
        shared.METHODS = (*shared.METHODS, "deepinteract")
    shared.load_or_build_dataset = load_dataset
    shared.run_epoch = run_epoch
