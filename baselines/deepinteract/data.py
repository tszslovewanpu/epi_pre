"""Independent full-chain geometry context for the shared AsEP samples.

The PLM cache and se3bench loader are read only. Full SEQRES positions, including
unresolved residues, remain in this context; selected_indices maps exactly back
to the shared CDR/surface order. Atom order is N, CA, C, CB. Missing atoms have
zero coordinates and a false mask, never invented or interpolated coordinates.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from baselines.se3bench import data as shared


CONTEXT_VERSION = 1
ATOM_NAMES = ("N", "CA", "C", "CB")
MAX_PEPTIDE_BOND_ANGSTROM = 2.2
CONTEXT_POLICY = {
    "atom_order": list(ATOM_NAMES),
    "alignment": "shared SEQRES-to-PDB global alignment, physical chains kept separate",
    "continuity": "adjacent SEQRES and mapped ATOM residues in one chain, finite C-N distance in (0.5, 2.2] Angstrom",
    "missing_atoms": "zero with explicit atom mask; no random replacement or interpolation",
    "glycine_cb": "always invalid; no virtual CB",
    "selected_validation": "exact N/CA/C coordinates and shared coordinate-mask equality",
}


def _one_partner_context(pdb_chains, seqres, selected_mask) -> dict[str, Tensor]:
    length = sum(len(str(sequence)) for sequence in seqres.values())
    if len(selected_mask) != length:
        raise ValueError("selection mask and complete SEQRES lengths differ")
    coords = torch.zeros(length, 4, 3, dtype=torch.float32)
    atom_mask = torch.zeros(length, 4, dtype=torch.bool)
    chain_id = torch.full((length,), -1, dtype=torch.long)
    residue_index = torch.full((length,), -1, dtype=torch.long)
    atom_indices = torch.full((length,), -1, dtype=torch.long)
    offset = 0
    for chain_number, (chain, sequence) in enumerate(seqres.items()):
        sequence = str(sequence)
        residues = pdb_chains.get(str(chain), [])
        mapping = shared._build_seqres_to_atom_mapping(sequence, residues)
        chain_id[offset:offset + len(sequence)] = chain_number
        residue_index[offset:offset + len(sequence)] = torch.arange(len(sequence))
        for sequence_index, atom_index in mapping.items():
            residue = residues[atom_index]
            full_index = offset + sequence_index
            atom_indices[full_index] = atom_index
            for atom_number, atom_name in enumerate(ATOM_NAMES):
                if atom_name not in residue or (atom_name == "CB" and sequence[sequence_index] == "G"):
                    continue
                xyz = torch.as_tensor(np.asarray(residue[atom_name].coord, dtype=np.float32))
                if bool(torch.isfinite(xyz).all()):
                    coords[full_index, atom_number] = xyz
                    atom_mask[full_index, atom_number] = True
        offset += len(sequence)
    link_mask = (
        (chain_id[:-1] == chain_id[1:])
        & (residue_index[1:] == residue_index[:-1] + 1)
        & (atom_indices[:-1] >= 0)
        & (atom_indices[1:] == atom_indices[:-1] + 1)
        & atom_mask[:-1, 2]
        & atom_mask[1:, 0]
    )
    cn_distance = torch.linalg.vector_norm(coords[:-1, 2] - coords[1:, 0], dim=-1)
    link_mask &= (cn_distance > .5) & (cn_distance <= MAX_PEPTIDE_BOND_ANGSTROM)
    selected_indices = torch.as_tensor(np.asarray(selected_mask, dtype=bool)).nonzero().flatten()
    return {
        "coords": coords,
        "atom_mask": atom_mask,
        "chain_id": chain_id,
        "residue_index": residue_index,
        "link_mask": link_mask,
        "selected_indices": selected_indices,
        "selected_coord_mask": atom_mask[selected_indices, :3].all(dim=-1),
    }


def validate_context_against_sample(context: Mapping[str, Tensor], sample, partner: str) -> None:
    """Fail if the new mapping is not exactly the shared selected-atom mapping."""
    indices = context["selected_indices"]
    valid = context["selected_coord_mask"]
    expected_mask = getattr(sample, f"coord_mask_{partner}").cpu()
    if not torch.equal(valid.cpu(), expected_mask):
        raise ValueError(f"{sample.abdbid}/{partner}: full-context/shared coordinate masks differ")
    for atom_number, atom_name in enumerate(("n", "ca", "c")):
        actual = context["coords"][indices, atom_number].masked_fill(~valid[:, None], 0.).cpu()
        expected = getattr(sample, f"coords_{atom_name}_{partner}").cpu()
        if not torch.equal(actual, expected):
            raise ValueError(f"{sample.abdbid}/{partner}/{atom_name}: full-context/shared coordinates differ")


def make_pair_context(sample, interim_graph_dir, pdb_root) -> dict[str, dict[str, Tensor]]:
    interim = shared.load_interim_graph(sample.abdbid, interim_graph_dir)
    pdb_path = Path(pdb_root).expanduser().resolve() / f"{sample.abdbid}.pdb"
    if not pdb_path.is_file():
        raise FileNotFoundError(pdb_path)
    pdb_chains = shared._parse_pdb_chains(pdb_path)
    result = {}
    for partner, mask_name in (("ab", "seqres2cdr"), ("ag", "seqres2surf")):
        context = _one_partner_context(
            pdb_chains, interim["seqres"][partner],
            shared._as_bool_numpy(interim["mapping"][partner][mask_name]),
        )
        validate_context_against_sample(context, sample, partner)
        result[partner] = context
    return result


def batch_context(contexts: Sequence[Mapping[str, Tensor]], device=None) -> dict[str, Tensor]:
    """Pad CPU contexts into a batch; -1 selected_indices denotes padding."""
    if not contexts:
        raise ValueError("cannot batch an empty context list")
    full_length = max(context["coords"].shape[0] for context in contexts)
    selected_length = max(context["selected_indices"].numel() for context in contexts)
    batch = len(contexts)
    output = {
        "coords": torch.zeros(batch, full_length, 4, 3, dtype=contexts[0]["coords"].dtype),
        "atom_mask": torch.zeros(batch, full_length, 4, dtype=torch.bool),
        "chain_id": torch.full((batch, full_length), -1, dtype=torch.long),
        "residue_index": torch.full((batch, full_length), -1, dtype=torch.long),
        "link_mask": torch.zeros(batch, max(full_length - 1, 0), dtype=torch.bool),
        "selected_indices": torch.full((batch, selected_length), -1, dtype=torch.long),
        "selected_coord_mask": torch.zeros(batch, selected_length, dtype=torch.bool),
    }
    for row, context in enumerate(contexts):
        for name in output:
            value = context[name].cpu()
            output[name][row, :value.shape[0]] = value
    return {name: value.to(device=device) for name, value in output.items()}


def jitter_context(context: Mapping[str, Tensor], std: float, generator=None) -> dict[str, Tensor]:
    """Perturb valid full N/CA/C once; CB stays fixed by the shared protocol.

    Accepts a batched or unbatched context. The original tensors are not changed.
    Selected coordinates must be gathered from the returned context, so selected
    nodes and their unselected frame-support atoms receive consistent noise.
    Original physical-chain connectivity is retained under augmentation.
    """
    if not np.isfinite(std) or std < 0:
        raise ValueError("jitter standard deviation must be finite and non-negative")
    output = {name: value.clone() for name, value in context.items()}
    if std:
        coords = output["coords"]
        mask = output["atom_mask"][..., :3, None]
        noise = torch.randn(coords[..., :3, :].shape, dtype=coords.dtype, device=coords.device, generator=generator)
        coords[..., :3, :] += noise * float(std) * mask.to(coords.dtype)
    return output


def selected_coordinates(context: Mapping[str, Tensor]) -> dict[str, Tensor]:
    """Gather aligned selected N/CA/C tensors and their common backbone mask."""
    indices = context["selected_indices"]
    if indices.ndim != 2:
        raise ValueError("selected_coordinates requires batched context")
    rows = torch.arange(indices.shape[0], device=indices.device)[:, None]
    valid = (indices >= 0) & context["selected_coord_mask"]
    selected = context["coords"][rows, indices.clamp_min(0)]
    selected = selected.masked_fill(~valid[..., None, None], 0.)
    return {"coords_n": selected[..., 0, :], "coords_ca": selected[..., 1, :],
            "coords_c": selected[..., 2, :], "coord_mask": valid}


def _fingerprint(samples, config) -> tuple[str, dict[str, Any]]:
    graph_root = Path(config["interim_graph_dir"]).expanduser().resolve()
    pdb_root = Path(config["pdb_root"]).expanduser().resolve()
    input_digest = hashlib.sha256()
    shared_coords_digest = hashlib.sha256()
    for sample in samples:
        input_digest.update(sample.abdbid.encode())
        for source in (graph_root / f"{sample.abdbid}.pt", pdb_root / f"{sample.abdbid}.pdb"):
            stat = source.stat()
            input_digest.update(f"{source}:{stat.st_size}:{stat.st_mtime_ns}".encode())
            if source.suffix == ".pdb":
                input_digest.update(hashlib.sha256(source.read_bytes()).digest())
        shared_coords_digest.update(sample.abdbid.encode())
        for partner in ("ab", "ag"):
            for name in ("coords_n", "coords_ca", "coords_c", "coord_mask"):
                value = getattr(sample, f"{name}_{partner}").detach().cpu().contiguous()
                shared_coords_digest.update(value.numpy().tobytes())
    payload = {
        "schema_version": CONTEXT_VERSION,
        "num_samples": len(samples),
        "interim_graph_dir": str(graph_root),
        "pdb_root": str(pdb_root),
        "source_manifest_sha256": input_digest.hexdigest(),
        "shared_selected_coordinates_sha256": shared_coords_digest.hexdigest(),
        "context_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "shared_data_code_sha256": hashlib.sha256(Path(shared.__file__).read_bytes()).hexdigest(),
        "policy": CONTEXT_POLICY,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(), payload


def load_or_build_context_cache(data_list, data_config: Mapping[str, Any], cache_path):
    """Return (contexts_by_abdbid, metadata); reject stale caches explicitly."""
    path = Path(cache_path).expanduser().resolve()
    fingerprint, payload = _fingerprint(data_list, data_config)
    if path.exists():
        cached = shared.safe_torch_load(path)
        if cached.get("fingerprint") != fingerprint:
            raise ValueError(f"stale DeepInteract geometry context cache: {path}; use a new cache path or explicitly remove this cache")
        contexts = cached["contexts"]
        if set(contexts) != {sample.abdbid for sample in data_list}:
            raise ValueError("cached context IDs differ from shared dataset")
        for sample in data_list:
            for partner in ("ab", "ag"):
                validate_context_against_sample(contexts[sample.abdbid][partner], sample, partner)
        digest = hashlib.sha256()
        for sample in data_list:
            for partner in ("ab", "ag"):
                for name, tensor in contexts[sample.abdbid][partner].items():
                    digest.update(name.encode())
                    digest.update(tensor.contiguous().numpy().tobytes())
        if digest.hexdigest() != cached["metadata"].get("coordinates_sha256"):
            raise ValueError(f"DeepInteract context coordinate checksum mismatch: {path}")
        return contexts, dict(cached["metadata"], loaded_from_cache=True, cache_path=str(path))
    contexts = {}
    coordinate_digest = hashlib.sha256()
    max_full_length = 0
    max_selected_index = -1
    totals = {"residues": 0, "selected_residues": 0, "valid_backbones": 0, "valid_cb": 0, "continuous_links": 0}
    for number, sample in enumerate(data_list):
        pair = make_pair_context(sample, data_config["interim_graph_dir"], data_config["pdb_root"])
        contexts[sample.abdbid] = pair
        for context in pair.values():
            max_full_length = max(max_full_length, context["coords"].shape[0])
            if context["selected_indices"].numel():
                max_selected_index = max(max_selected_index, int(context["selected_indices"].max()))
            for name, tensor in context.items():
                coordinate_digest.update(name.encode())
                coordinate_digest.update(tensor.contiguous().numpy().tobytes())
            totals["residues"] += context["coords"].shape[0]
            totals["selected_residues"] += context["selected_indices"].numel()
            totals["valid_backbones"] += int(context["atom_mask"][:, :3].all(-1).sum())
            totals["valid_cb"] += int(context["atom_mask"][:, 3].sum())
            totals["continuous_links"] += int(context["link_mask"].sum())
        if bool(data_config.get("verbose", True)) and ((number + 1) % 100 == 0 or number + 1 == len(data_list)):
            print(f"DeepInteract full-chain contexts: {number + 1}/{len(data_list)}", flush=True)
    metadata = {
        **payload, "fingerprint": fingerprint,
        "coordinates_sha256": coordinate_digest.hexdigest(),
        "max_full_length": max_full_length,
        "max_selected_global_index": max_selected_index,
        "minimum_node_embedding_capacity": max_selected_index + 1,
        "counts": totals,
        "loaded_from_cache": False,
        "cache_path": str(path),
    }
    shared._atomic_torch_save({"fingerprint": fingerprint, "contexts": contexts, "metadata": metadata}, path)
    return contexts, metadata
