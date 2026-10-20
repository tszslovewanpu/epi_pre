"""Standalone AsEP data loading for the SE(3) geometry baselines.

This module deliberately has no dependency on :mod:`torch_geometric`.  It
loads the official AsEP interim graphs, selects antibody CDR and antigen
surface residues, aligns backbone coordinates from the matching PDB file, and
returns ordinary picklable :class:`PairSample` objects.

Dataset paths are explicit.  A typical configuration is::

    {
        "id_list_path": ".../asepv1-AbDb-IDs.txt",
        "interim_graph_dir": ".../asepv1_interim_graphs",
        "pdb_root": ".../structures",
        "split_path": ".../split_dict.pt",
        "split_method": "epitope_ratio",
        "cache_path": ".../asep_plm_onehot_v1.pt",
        "ab_embedding_model": "igfold",
        "ag_embedding_model": "esm2",
        "include_aux_features": True,
        "strict_coords": False,
    }

Only the data list and its schema/configuration fingerprint are cached.  Split
indices are always loaded independently from the official ``split_dict.pt`` so
one feature cache can safely serve both ratio and group experiments.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from Bio.Align import PairwiseAligner
from Bio.PDB import PDBParser
from torch import Tensor

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - tqdm is optional for library use.
    tqdm = None


CACHE_SCHEMA = "asep_se3bench_pairs_v1"
AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
    "MSE": "M",
    "SEC": "U",
    "PYL": "O",
}
AA1_TO_IDX = {aa: i for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")}
NO_PLM_NAMES = {"none", "no_plm", "no-plm", "zero", "zeros", "null"}
SPLIT_ALIASES = {
    "ratio": "epitope_ratio",
    "epitope_ratio": "epitope_ratio",
    "group": "epitope_group",
    "epitope_group": "epitope_group",
}


class CacheFingerprintError(ValueError):
    """Raised when a cache was built from a different dataset configuration."""


@dataclass
class PairSample:
    """One antibody-antigen complex represented by ordinary CPU tensors."""

    x_b: Tensor
    x_g: Tensor
    aux_b: Tensor
    aux_g: Tensor
    y_g: Tensor
    coords_ca_ab: Tensor
    coords_n_ab: Tensor
    coords_c_ab: Tensor
    coord_mask_ab: Tensor
    coords_ca_ag: Tensor
    coords_n_ag: Tensor
    coords_c_ag: Tensor
    coord_mask_ag: Tensor
    cdr_type: Tensor
    mask_b: Tensor
    mask_g: Tensor
    edge_index_ab: Tensor
    edge_index_ag: Tensor
    abdbid: str

    def to(self, *args: Any, **kwargs: Any) -> "PairSample":
        """Return a copy whose tensor fields have been moved with ``Tensor.to``."""

        values: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.to(*args, **kwargs) if isinstance(value, Tensor) else value
        return type(self)(**values)

    def pin_memory(self) -> "PairSample":
        """Return a copy with all tensor fields placed in pinned CPU memory."""

        values: Dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            values[field.name] = value.pin_memory() if isinstance(value, Tensor) else value
        return type(self)(**values)


def safe_torch_load(path: os.PathLike[str] | str) -> Any:
    """Load a trusted local torch artifact on CPU across torch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only argument.
        return torch.load(path, map_location="cpu")


def _atomic_torch_save(obj: Any, path: Path) -> None:
    """Write a torch artifact atomically using a temporary file beside it."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        torch.save(obj, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _as_float_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu().float().contiguous()
    return torch.as_tensor(value, dtype=torch.float32).contiguous()


def _as_long_tensor(value: Any) -> Tensor:
    if isinstance(value, Tensor):
        return value.detach().cpu().long().contiguous()
    return torch.as_tensor(value, dtype=torch.long).contiguous()


def _as_bool_numpy(value: Any) -> np.ndarray:
    if isinstance(value, Tensor):
        return value.detach().cpu().numpy().astype(bool, copy=False)
    return np.asarray(value).astype(bool, copy=False)


def _canonical_path(path: os.PathLike[str] | str) -> Path:
    return Path(path).expanduser().resolve()


def read_asep_ids(id_list_path: os.PathLike[str] | str) -> List[str]:
    """Read and validate the canonical AsEP sample ordering."""

    path = _canonical_path(id_list_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        identifiers = [line.strip() for line in handle if line.strip()]
    if not identifiers:
        raise ValueError(f"no AsEP IDs found in {path}")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError(f"duplicate AsEP IDs found in {path}")
    return identifiers


def _source_manifest_hash(
    identifiers: Sequence[str],
    id_list_path: Path,
    interim_graph_dir: Path,
    pdb_root: Path,
) -> str:
    """Hash source identities without reading gigabytes of tensor/PDB contents."""

    digest = hashlib.sha256()
    digest.update(id_list_path.read_bytes())
    for abdbid in identifiers:
        for kind, path in (
            ("graph", interim_graph_dir / f"{abdbid}.pt"),
            ("pdb", pdb_root / f"{abdbid}.pdb"),
        ):
            digest.update(kind.encode("ascii"))
            digest.update(abdbid.encode("utf-8"))
            if not path.is_file():
                digest.update(b"MISSING")
                continue
            stat = path.stat()
            digest.update(str(stat.st_size).encode("ascii"))
            digest.update(str(stat.st_mtime_ns).encode("ascii"))
    return digest.hexdigest()


def dataset_fingerprint(
    *,
    id_list_path: os.PathLike[str] | str,
    interim_graph_dir: os.PathLike[str] | str,
    pdb_root: os.PathLike[str] | str,
    emb_model_ab: str = "igfold",
    emb_model_ag: str = "esm2",
    include_aux_features: bool = True,
    normalize_plm: bool = False,
    strict_coords: bool = False,
) -> Tuple[str, Dict[str, Any]]:
    """Return the cache fingerprint and the canonical payload that produced it."""

    id_path = _canonical_path(id_list_path)
    graph_dir = _canonical_path(interim_graph_dir)
    pdb_dir = _canonical_path(pdb_root)
    if not graph_dir.is_dir():
        raise NotADirectoryError(graph_dir)
    if not pdb_dir.is_dir():
        raise NotADirectoryError(pdb_dir)
    identifiers = read_asep_ids(id_path)
    missing_graphs = [abdbid for abdbid in identifiers if not (graph_dir / f"{abdbid}.pt").is_file()]
    if missing_graphs:
        preview = ", ".join(missing_graphs[:5])
        raise FileNotFoundError(f"{len(missing_graphs)} interim graphs are missing from {graph_dir}: {preview}")

    payload: Dict[str, Any] = {
        "schema": CACHE_SCHEMA,
        "id_list_path": str(id_path),
        "interim_graph_dir": str(graph_dir),
        "pdb_root": str(pdb_dir),
        "emb_model_ab": str(emb_model_ab).lower(),
        "emb_model_ag": str(emb_model_ag).lower(),
        "include_aux_features": bool(include_aux_features),
        "aux_feature_mode": "one_hot" if include_aux_features else "none",
        "normalize_plm": bool(normalize_plm),
        "strict_coords": bool(strict_coords),
        "source_manifest_hash": _source_manifest_hash(identifiers, id_path, graph_dir, pdb_dir),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest(), payload


def _maybe_zscore_tokens(x: Tensor, enabled: bool) -> Tensor:
    if not enabled or x.numel() == 0 or x.size(0) <= 1:
        return x
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def _select_by_mask_or_validate(embedding: Any, mask: np.ndarray, name: str, abdbid: str) -> Tensor:
    x = _as_float_tensor(embedding)
    mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
    if x.ndim != 2:
        raise ValueError(f"{abdbid}: {name} embedding must be rank 2, got {tuple(x.shape)}")
    if x.size(0) == mask_tensor.numel():
        return x[mask_tensor]
    if x.size(0) == int(mask_tensor.sum().item()):
        return x
    raise ValueError(
        f"{abdbid}: cannot align {name} embedding length {x.size(0)} with "
        f"mask length {mask_tensor.numel()} / selected {int(mask_tensor.sum().item())}"
    )


def _one_hot_from_seqres(seqres_dict: Mapping[str, str], flat_mask: np.ndarray) -> Tensor:
    sequence = "".join(str(sequence) for sequence in seqres_dict.values())
    if len(sequence) != len(flat_mask):
        raise ValueError(f"seqres length {len(sequence)} != mask length {len(flat_mask)}")
    features = np.zeros((len(sequence), 20), dtype=np.float32)
    for index, residue in enumerate(sequence):
        aa_index = AA1_TO_IDX.get(residue)
        if aa_index is not None:
            features[index, aa_index] = 1.0
    return torch.from_numpy(features[flat_mask]).float()


def _load_base_features(
    interim: Mapping[str, Any],
    chain_key: str,
    embedding_model: str,
    seqres_dict: Mapping[str, str],
    flat_mask: np.ndarray,
    abdbid: str,
) -> Tensor:
    model = str(embedding_model or "none").lower()
    if model in NO_PLM_NAMES:
        return torch.zeros(int(flat_mask.sum()), 1, dtype=torch.float32)
    if model == "one_hot":
        return _one_hot_from_seqres(seqres_dict, flat_mask)
    try:
        embedding = interim["embedding"][chain_key][model]
    except KeyError as error:
        available = sorted(interim.get("embedding", {}).get(chain_key, {}).keys())
        raise KeyError(
            f"{abdbid}: embedding {model!r} is unavailable for {chain_key}; available={available}"
        ) from error
    return _select_by_mask_or_validate(embedding, flat_mask, f"{chain_key}/{model}", abdbid)


def _residue_to_aa1(residue: Any) -> str:
    return AA3_TO_AA1.get(residue.get_resname().upper(), "X")


def _has_backbone(residue: Any) -> bool:
    return "N" in residue and "CA" in residue and "C" in residue


def _parse_pdb_chains(pdb_path: Path) -> Dict[str, List[Any]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", str(pdb_path))
    model = next(iter(structure), None)
    if model is None:
        return {}
    chains: Dict[str, List[Any]] = {}
    for chain in model:
        chains[str(chain.id)] = [
            residue
            for residue in chain
            if residue.id[0] == " " and _residue_to_aa1(residue) != "X"
        ]
    return chains


def _build_seqres_to_atom_mapping(seqres: str, atom_residues: Sequence[Any]) -> Dict[int, int]:
    sequence = str(seqres)
    atom_sequence = "".join(_residue_to_aa1(residue) for residue in atom_residues)
    if not sequence or not atom_sequence:
        return {}

    aligner = PairwiseAligner()
    aligner.mode = "global"
    aligner.match_score = 2.0
    aligner.mismatch_score = -1.0
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -1.0
    alignment = aligner.align(sequence, atom_sequence)[0]

    mapping: Dict[int, int] = {}
    for (seq_start, seq_end), (atom_start, atom_end) in zip(alignment.aligned[0], alignment.aligned[1]):
        for offset in range(min(seq_end - seq_start, atom_end - atom_start)):
            mapping[int(seq_start + offset)] = int(atom_start + offset)
    return mapping


def _extract_chain_coords_by_mask(
    atom_residues: Sequence[Any],
    seqres: str,
    select_mask: np.ndarray,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    mapping = _build_seqres_to_atom_mapping(seqres, atom_residues)
    ca_list: List[np.ndarray] = []
    n_list: List[np.ndarray] = []
    c_list: List[np.ndarray] = []
    valid_list: List[bool] = []

    for seq_index, selected in enumerate(select_mask):
        if not selected:
            continue
        atom_index = mapping.get(seq_index)
        if atom_index is not None and atom_index < len(atom_residues) and _has_backbone(atom_residues[atom_index]):
            residue = atom_residues[atom_index]
            ca_list.append(np.asarray(residue["CA"].coord, dtype=np.float32))
            n_list.append(np.asarray(residue["N"].coord, dtype=np.float32))
            c_list.append(np.asarray(residue["C"].coord, dtype=np.float32))
            valid_list.append(True)
        else:
            ca_list.append(np.zeros(3, dtype=np.float32))
            n_list.append(np.zeros(3, dtype=np.float32))
            c_list.append(np.zeros(3, dtype=np.float32))
            valid_list.append(False)

    if not ca_list:
        empty = torch.empty(0, 3, dtype=torch.float32)
        return empty, empty.clone(), empty.clone(), torch.empty(0, dtype=torch.bool)
    return (
        torch.from_numpy(np.stack(ca_list)).float(),
        torch.from_numpy(np.stack(n_list)).float(),
        torch.from_numpy(np.stack(c_list)).float(),
        torch.as_tensor(valid_list, dtype=torch.bool),
    )


def _split_flat_mask_by_chains(
    seqres_dict: Mapping[str, str],
    flat_mask: np.ndarray,
) -> Dict[str, np.ndarray]:
    masks: Dict[str, np.ndarray] = {}
    offset = 0
    for chain_id, sequence in seqres_dict.items():
        length = len(sequence)
        masks[str(chain_id)] = np.asarray(flat_mask[offset : offset + length]).astype(bool, copy=False)
        offset += length
    if offset != len(flat_mask):
        raise ValueError(f"flat mask length {len(flat_mask)} != concatenated seqres length {offset}")
    return masks


def _extract_multichain_coords(
    pdb_chains: Mapping[str, Sequence[Any]],
    seqres_dict: Mapping[str, str],
    flat_select_mask: np.ndarray,
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    ca_all: List[Tensor] = []
    n_all: List[Tensor] = []
    c_all: List[Tensor] = []
    valid_all: List[Tensor] = []
    masks = _split_flat_mask_by_chains(seqres_dict, flat_select_mask)
    for chain_id, sequence in seqres_dict.items():
        ca, n, c, valid = _extract_chain_coords_by_mask(
            pdb_chains.get(str(chain_id), []),
            sequence,
            masks[str(chain_id)],
        )
        ca_all.append(ca)
        n_all.append(n)
        c_all.append(c)
        valid_all.append(valid)
    return (
        torch.cat(ca_all, dim=0),
        torch.cat(n_all, dim=0),
        torch.cat(c_all, dim=0),
        torch.cat(valid_all, dim=0),
    )


def _infer_cdr_types(seqres_ab: Mapping[str, str], cdr_mask_flat: np.ndarray) -> Tensor:
    cdr_types: List[int] = []
    masks = _split_flat_mask_by_chains(seqres_ab, cdr_mask_flat)
    for chain_id, mask in masks.items():
        segments: List[Tuple[int, int]] = []
        index = 0
        while index < len(mask):
            if not mask[index]:
                index += 1
                continue
            end = index
            while end < len(mask) and mask[end]:
                end += 1
            segments.append((index, end))
            index = end

        chain_upper = chain_id.upper()
        base = 0 if chain_upper == "H" else 3 if chain_upper == "L" else 6
        type_by_position = np.full(len(mask), 6, dtype=np.int64)
        for segment_index, (start, end) in enumerate(segments[:3]):
            type_by_position[start:end] = 6 if base == 6 else base + segment_index
        cdr_types.extend(int(type_by_position[position]) for position, selected in enumerate(mask) if selected)
    return torch.as_tensor(cdr_types, dtype=torch.long)


def _get_node_labels(interim: Mapping[str, Any], surface_mask: np.ndarray, ng: int, abdbid: str) -> Tensor:
    try:
        epitope_mapping = interim["mapping"]["ag"]["seqres2epitope"]
    except KeyError as error:
        raise KeyError(f"{abdbid}: missing mapping.ag.seqres2epitope labels") from error
    epitope_mask = _as_bool_numpy(epitope_mapping)
    if epitope_mask.size != surface_mask.size:
        raise ValueError(
            f"{abdbid}: epitope mask length {epitope_mask.size} != surface mask length {surface_mask.size}"
        )
    labels = torch.as_tensor(epitope_mask[surface_mask], dtype=torch.float32)
    if labels.numel() != ng:
        raise ValueError(f"{abdbid}: label length {labels.numel()} != antigen node count {ng}")
    return labels


def _get_intra_chain_edges(interim: Mapping[str, Any], chain_key: str, n_nodes: int, abdbid: str) -> Tensor:
    try:
        adjacency = interim["edges"][chain_key]
    except KeyError as error:
        raise KeyError(f"{abdbid}: missing intra-chain edge tensor for {chain_key!r}") from error

    if isinstance(adjacency, Tensor) and adjacency.layout == torch.sparse_coo:
        sparse = adjacency.coalesce()
        if tuple(sparse.size()) != (n_nodes, n_nodes):
            raise ValueError(
                f"{abdbid}: {chain_key} adjacency shape {tuple(sparse.size())} != {(n_nodes, n_nodes)}"
            )
        edge_index = sparse.indices()
    elif isinstance(adjacency, Tensor) and adjacency.ndim == 2 and adjacency.size(0) == 2:
        edge_index = adjacency
    else:
        raise TypeError(f"{abdbid}: unsupported {chain_key} edge representation {type(adjacency)!r}")

    edge_index = edge_index.detach().cpu().long().contiguous()
    if edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError(f"{abdbid}: {chain_key} edge_index must have shape [2,E]")
    if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= n_nodes):
        raise ValueError(f"{abdbid}: {chain_key} edge_index is outside [0,{n_nodes})")
    return edge_index


def load_interim_graph(abdbid: str, interim_graph_dir: os.PathLike[str] | str) -> Mapping[str, Any]:
    path = _canonical_path(interim_graph_dir) / f"{abdbid}.pt"
    if not path.is_file():
        raise FileNotFoundError(path)
    graph = safe_torch_load(path)
    if not isinstance(graph, Mapping):
        raise TypeError(f"{abdbid}: interim graph must be a mapping, got {type(graph)!r}")
    return graph


def make_pair_sample(
    *,
    abdbid: str,
    interim: Mapping[str, Any],
    pdb_root: os.PathLike[str] | str,
    emb_model_ab: str = "igfold",
    emb_model_ag: str = "esm2",
    include_aux_features: bool = True,
    normalize_plm: bool = False,
    strict_coords: bool = False,
) -> PairSample:
    """Build one validated pair sample from an interim graph and its PDB.

    ``strict_coords`` requires the PDB file to exist.  Individual unresolved
    residues remain represented by zero coordinates and a false coordinate
    mask, matching the source LF3DRoPE loader.
    """

    cdr_mask = _as_bool_numpy(interim["mapping"]["ab"]["seqres2cdr"])
    surface_mask = _as_bool_numpy(interim["mapping"]["ag"]["seqres2surf"])
    seqres_ab: Mapping[str, str] = interim["seqres"]["ab"]
    seqres_ag: Mapping[str, str] = interim["seqres"]["ag"]

    if cdr_mask.size != sum(len(sequence) for sequence in seqres_ab.values()):
        raise ValueError(f"{abdbid}: antibody sequence/CDR-mask length mismatch")
    if surface_mask.size != sum(len(sequence) for sequence in seqres_ag.values()):
        raise ValueError(f"{abdbid}: antigen sequence/surface-mask length mismatch")

    x_b = _maybe_zscore_tokens(
        _load_base_features(interim, "ab", emb_model_ab, seqres_ab, cdr_mask, abdbid),
        normalize_plm,
    )
    x_g = _maybe_zscore_tokens(
        _load_base_features(interim, "ag", emb_model_ag, seqres_ag, surface_mask, abdbid),
        normalize_plm,
    )

    aux_b = torch.empty((x_b.size(0), 0), dtype=torch.float32)
    aux_g = torch.empty((x_g.size(0), 0), dtype=torch.float32)
    if include_aux_features:
        aux_b = _one_hot_from_seqres(seqres_ab, cdr_mask)
        aux_g = _one_hot_from_seqres(seqres_ag, surface_mask)
        x_b = torch.cat([x_b, aux_b], dim=-1)
        x_g = torch.cat([x_g, aux_g], dim=-1)

    y_g = _get_node_labels(interim, surface_mask, x_g.size(0), abdbid)
    cdr_type = _infer_cdr_types(seqres_ab, cdr_mask)
    edge_index_ab = _get_intra_chain_edges(interim, "ab", x_b.size(0), abdbid)
    edge_index_ag = _get_intra_chain_edges(interim, "ag", x_g.size(0), abdbid)

    pdb_path = _canonical_path(pdb_root) / f"{abdbid}.pdb"
    if pdb_path.is_file():
        pdb_chains = _parse_pdb_chains(pdb_path)
        coords_ab = _extract_multichain_coords(pdb_chains, seqres_ab, cdr_mask)
        coords_ag = _extract_multichain_coords(pdb_chains, seqres_ag, surface_mask)
    else:
        if strict_coords:
            raise FileNotFoundError(pdb_path)
        nb, ng = x_b.size(0), x_g.size(0)
        coords_ab = (
            torch.zeros((nb, 3)),
            torch.zeros((nb, 3)),
            torch.zeros((nb, 3)),
            torch.zeros(nb, dtype=torch.bool),
        )
        coords_ag = (
            torch.zeros((ng, 3)),
            torch.zeros((ng, 3)),
            torch.zeros((ng, 3)),
            torch.zeros(ng, dtype=torch.bool),
        )

    sample = PairSample(
        x_b=x_b.float().contiguous(),
        x_g=x_g.float().contiguous(),
        aux_b=aux_b.float().contiguous(),
        aux_g=aux_g.float().contiguous(),
        y_g=y_g.float().contiguous(),
        coords_ca_ab=coords_ab[0].float().contiguous(),
        coords_n_ab=coords_ab[1].float().contiguous(),
        coords_c_ab=coords_ab[2].float().contiguous(),
        coord_mask_ab=coords_ab[3].bool().contiguous(),
        coords_ca_ag=coords_ag[0].float().contiguous(),
        coords_n_ag=coords_ag[1].float().contiguous(),
        coords_c_ag=coords_ag[2].float().contiguous(),
        coord_mask_ag=coords_ag[3].bool().contiguous(),
        cdr_type=cdr_type.long().contiguous(),
        mask_b=torch.ones(x_b.size(0), dtype=torch.bool),
        mask_g=torch.ones(x_g.size(0), dtype=torch.bool),
        edge_index_ab=edge_index_ab,
        edge_index_ag=edge_index_ag,
        abdbid=str(abdbid),
    )
    validate_pair_sample(sample)
    return sample


def validate_pair_sample(sample: PairSample) -> None:
    """Raise a descriptive error if one sample has inconsistent tensor shapes."""

    if not isinstance(sample, PairSample):
        raise TypeError(f"expected PairSample, got {type(sample)!r}")
    nb, ng = sample.x_b.size(0), sample.x_g.size(0)
    if sample.x_b.ndim != 2 or sample.x_g.ndim != 2 or nb == 0 or ng == 0:
        raise ValueError(f"{sample.abdbid}: x_b/x_g must be non-empty rank-2 tensors")

    ab_vectors = (sample.coords_ca_ab, sample.coords_n_ab, sample.coords_c_ab)
    ag_vectors = (sample.coords_ca_ag, sample.coords_n_ag, sample.coords_c_ag)
    if any(tuple(tensor.shape) != (nb, 3) for tensor in ab_vectors):
        raise ValueError(f"{sample.abdbid}: antibody coordinate shape mismatch")
    if any(tuple(tensor.shape) != (ng, 3) for tensor in ag_vectors):
        raise ValueError(f"{sample.abdbid}: antigen coordinate shape mismatch")
    if tuple(sample.coord_mask_ab.shape) != (nb,) or tuple(sample.coord_mask_ag.shape) != (ng,):
        raise ValueError(f"{sample.abdbid}: coordinate-mask shape mismatch")
    if tuple(sample.mask_b.shape) != (nb,) or tuple(sample.mask_g.shape) != (ng,):
        raise ValueError(f"{sample.abdbid}: token-mask shape mismatch")
    if tuple(sample.y_g.shape) != (ng,) or tuple(sample.cdr_type.shape) != (nb,):
        raise ValueError(f"{sample.abdbid}: label/CDR-type shape mismatch")
    if sample.aux_b.shape[:1] != (nb,) or sample.aux_g.shape[:1] != (ng,):
        raise ValueError(f"{sample.abdbid}: auxiliary-feature shape mismatch")

    if not torch.isfinite(sample.x_b).all() or not torch.isfinite(sample.x_g).all():
        raise ValueError(f"{sample.abdbid}: non-finite residue features")
    for tensor in (*ab_vectors, *ag_vectors):
        if not torch.isfinite(tensor).all():
            raise ValueError(f"{sample.abdbid}: non-finite coordinates")
    if not torch.all((sample.y_g == 0) | (sample.y_g == 1)):
        raise ValueError(f"{sample.abdbid}: labels must be binary")
    if sample.cdr_type.numel() and (int(sample.cdr_type.min()) < 0 or int(sample.cdr_type.max()) > 6):
        raise ValueError(f"{sample.abdbid}: CDR types must be in [0,6]")

    for name, edge_index, node_count in (
        ("edge_index_ab", sample.edge_index_ab, nb),
        ("edge_index_ag", sample.edge_index_ag, ng),
    ):
        if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
            raise ValueError(f"{sample.abdbid}: {name} must be long [2,E]")
        if edge_index.numel() and (int(edge_index.min()) < 0 or int(edge_index.max()) >= node_count):
            raise ValueError(f"{sample.abdbid}: {name} contains an invalid node index")


def summarize_data_list(data_list: Sequence[PairSample]) -> Dict[str, Any]:
    """Validate a data list and return JSON-serializable aggregate metadata."""

    if not data_list:
        raise ValueError("data_list is empty")
    seen: set[str] = set()
    input_ab_dims: set[int] = set()
    input_ag_dims: set[int] = set()
    aux_ab_dims: set[int] = set()
    aux_ag_dims: set[int] = set()
    total_ab = total_ag = valid_ab = valid_ag = positives = 0
    edges_ab = edges_ag = 0
    samples_missing_ab = samples_missing_ag = samples_missing_any = 0

    for sample in data_list:
        validate_pair_sample(sample)
        if sample.abdbid in seen:
            raise ValueError(f"duplicate cached sample ID {sample.abdbid}")
        seen.add(sample.abdbid)
        input_ab_dims.add(int(sample.x_b.size(1)))
        input_ag_dims.add(int(sample.x_g.size(1)))
        aux_ab_dims.add(int(sample.aux_b.size(1)))
        aux_ag_dims.add(int(sample.aux_g.size(1)))
        total_ab += int(sample.x_b.size(0))
        total_ag += int(sample.x_g.size(0))
        valid_ab += int(sample.coord_mask_ab.sum().item())
        valid_ag += int(sample.coord_mask_ag.sum().item())
        missing_ab = not bool(sample.coord_mask_ab.all())
        missing_ag = not bool(sample.coord_mask_ag.all())
        samples_missing_ab += int(missing_ab)
        samples_missing_ag += int(missing_ag)
        samples_missing_any += int(missing_ab or missing_ag)
        positives += int(sample.y_g.sum().item())
        edges_ab += int(sample.edge_index_ab.size(1))
        edges_ag += int(sample.edge_index_ag.size(1))

    if len(input_ab_dims) != 1 or len(input_ag_dims) != 1:
        raise ValueError(f"inconsistent input dimensions: antibody={input_ab_dims}, antigen={input_ag_dims}")
    if len(aux_ab_dims) != 1 or len(aux_ag_dims) != 1:
        raise ValueError(f"inconsistent auxiliary dimensions: antibody={aux_ab_dims}, antigen={aux_ag_dims}")

    return {
        "num_samples": len(data_list),
        "input_ab_dim": next(iter(input_ab_dims)),
        "input_ag_dim": next(iter(input_ag_dims)),
        "aux_ab_dim": next(iter(aux_ab_dims)),
        "aux_ag_dim": next(iter(aux_ag_dims)),
        "total_ab_tokens": total_ab,
        "total_ag_tokens": total_ag,
        "positive_ag_tokens": positives,
        "valid_ab_coords": valid_ab,
        "valid_ag_coords": valid_ag,
        "samples_missing_ab_coords": samples_missing_ab,
        "samples_missing_ag_coords": samples_missing_ag,
        "samples_missing_any_coords": samples_missing_any,
        "ab_coord_coverage": valid_ab / max(total_ab, 1),
        "ag_coord_coverage": valid_ag / max(total_ag, 1),
        "total_ab_edges": edges_ab,
        "total_ag_edges": edges_ag,
    }


def build_data_list(
    *,
    id_list_path: os.PathLike[str] | str,
    interim_graph_dir: os.PathLike[str] | str,
    pdb_root: os.PathLike[str] | str,
    emb_model_ab: str = "igfold",
    emb_model_ag: str = "esm2",
    include_aux_features: bool = True,
    normalize_plm: bool = False,
    strict_coords: bool = False,
    verbose: bool = True,
) -> List[PairSample]:
    """Build all samples in the canonical ID-list order."""

    identifiers = read_asep_ids(id_list_path)
    graph_dir = _canonical_path(interim_graph_dir)
    pdb_dir = _canonical_path(pdb_root)
    iterator: Any = identifiers
    if verbose and tqdm is not None:
        iterator = tqdm(identifiers, desc="Building AsEP SE(3) baseline data")

    data_list: List[PairSample] = []
    for abdbid in iterator:
        data_list.append(
            make_pair_sample(
                abdbid=abdbid,
                interim=load_interim_graph(abdbid, graph_dir),
                pdb_root=pdb_dir,
                emb_model_ab=emb_model_ab,
                emb_model_ag=emb_model_ag,
                include_aux_features=include_aux_features,
                normalize_plm=normalize_plm,
                strict_coords=strict_coords,
            )
        )
    summarize_data_list(data_list)
    return data_list


def load_split_indices(
    split_path: os.PathLike[str] | str,
    split_method: str,
    *,
    num_samples: Optional[int] = None,
) -> Dict[str, Tensor]:
    """Load and validate one official AsEP split independently of the cache."""

    path = _canonical_path(split_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    method = SPLIT_ALIASES.get(str(split_method).lower())
    if method is None:
        raise ValueError(f"split_method must be one of {sorted(SPLIT_ALIASES)}, got {split_method!r}")
    split_object = safe_torch_load(path)
    if method not in split_object:
        raise KeyError(f"{path} does not contain split method {method!r}")
    raw_split = split_object[method]
    required = ("train", "val", "test")
    if any(name not in raw_split for name in required):
        raise KeyError(f"{path}:{method} must contain train, val, and test")

    split = {name: _as_long_tensor(raw_split[name]).view(-1) for name in required}
    for name, indices in split.items():
        if torch.unique(indices).numel() != indices.numel():
            raise ValueError(f"{path}:{method}:{name} contains duplicate indices")
        if indices.numel() and int(indices.min()) < 0:
            raise ValueError(f"{path}:{method}:{name} contains a negative index")
        if num_samples is not None and indices.numel() and int(indices.max()) >= num_samples:
            raise ValueError(f"{path}:{method}:{name} index exceeds data-list size {num_samples}")

    sets = {name: set(indices.tolist()) for name, indices in split.items()}
    if sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"]:
        raise ValueError(f"{path}:{method} split partitions overlap")
    if num_samples is not None:
        union = sets["train"] | sets["val"] | sets["test"]
        if union != set(range(num_samples)):
            raise ValueError(
                f"{path}:{method} covers {len(union)} indices but data_list contains {num_samples} samples"
            )
    return split


def _required_config_path(config: Mapping[str, Any], key: str) -> Path:
    value = config.get(key)
    if value is None or str(value).strip() == "":
        raise KeyError(f"dataset config requires explicit {key!r}")
    return _canonical_path(value)


def load_or_build_dataset(config: Mapping[str, Any]) -> Tuple[List[PairSample], Dict[str, Tensor], Dict[str, Any]]:
    """Load/build feature data, then independently load the requested split.

    Required configuration keys are ``id_list_path``, ``interim_graph_dir``,
    ``pdb_root``, and ``split_path``.  ``cache_path`` is optional.
    """

    id_list_path = _required_config_path(config, "id_list_path")
    interim_graph_dir = _required_config_path(config, "interim_graph_dir")
    pdb_root = _required_config_path(config, "pdb_root")
    split_path = _required_config_path(config, "split_path")
    split_method = str(config.get("split_method", "epitope_ratio"))
    ab_config = config.get("ab") if isinstance(config.get("ab"), Mapping) else {}
    ag_config = config.get("ag") if isinstance(config.get("ag"), Mapping) else {}
    emb_model_ab = str(
        config.get(
            "ab_embedding_model",
            config.get("emb_model_ab", ab_config.get("embedding_model", "igfold")),
        )
    )
    emb_model_ag = str(
        config.get(
            "ag_embedding_model",
            config.get("emb_model_ag", ag_config.get("embedding_model", "esm2")),
        )
    )
    include_aux_features = bool(config.get("include_aux_features", True))
    normalize_plm = bool(config.get("normalize_plm", False))
    strict_coords = bool(config.get("strict_coords", False))
    verbose = bool(config.get("verbose", True))
    rebuild_cache = bool(config.get("rebuild_cache", False))
    aux_mode = str(config.get("aux_feature_mode", "one_hot")).lower()
    if include_aux_features and aux_mode != "one_hot":
        raise ValueError("SE(3) baselines support only aux_feature_mode='one_hot'")

    fingerprint, fingerprint_payload = dataset_fingerprint(
        id_list_path=id_list_path,
        interim_graph_dir=interim_graph_dir,
        pdb_root=pdb_root,
        emb_model_ab=emb_model_ab,
        emb_model_ag=emb_model_ag,
        include_aux_features=include_aux_features,
        normalize_plm=normalize_plm,
        strict_coords=strict_coords,
    )

    cache_value = config.get("cache_path")
    cache_path = _canonical_path(cache_value) if cache_value else None
    loaded_from_cache = False
    if cache_path is not None and cache_path.is_file() and not rebuild_cache:
        cached = safe_torch_load(cache_path)
        if not isinstance(cached, Mapping):
            raise TypeError(f"cache {cache_path} must contain a mapping")
        actual_schema = cached.get("schema")
        actual_fingerprint = cached.get("fingerprint")
        if actual_schema != CACHE_SCHEMA or actual_fingerprint != fingerprint:
            raise CacheFingerprintError(
                f"cache fingerprint mismatch for {cache_path}; "
                f"expected schema={CACHE_SCHEMA!r}, fingerprint={fingerprint}, "
                f"found schema={actual_schema!r}, fingerprint={actual_fingerprint}. "
                "Use rebuild_cache=true to replace it."
            )
        data_list = cached.get("data_list")
        if not isinstance(data_list, list):
            raise TypeError(f"cache {cache_path} does not contain a data_list")
        loaded_from_cache = True
    else:
        data_list = build_data_list(
            id_list_path=id_list_path,
            interim_graph_dir=interim_graph_dir,
            pdb_root=pdb_root,
            emb_model_ab=emb_model_ab,
            emb_model_ag=emb_model_ag,
            include_aux_features=include_aux_features,
            normalize_plm=normalize_plm,
            strict_coords=strict_coords,
            verbose=verbose,
        )
        if cache_path is not None:
            _atomic_torch_save(
                {
                    "schema": CACHE_SCHEMA,
                    "fingerprint": fingerprint,
                    "fingerprint_payload": fingerprint_payload,
                    "data_list": data_list,
                },
                cache_path,
            )

    summary = summarize_data_list(data_list)
    identifiers = read_asep_ids(id_list_path)
    cached_ids = [sample.abdbid for sample in data_list]
    if cached_ids != identifiers:
        raise ValueError("data_list order does not exactly match id_list_path")

    split_idx = load_split_indices(split_path, split_method, num_samples=len(data_list))
    metadata: Dict[str, Any] = {
        "format": CACHE_SCHEMA,
        "fingerprint": fingerprint,
        "fingerprint_payload": fingerprint_payload,
        "loaded_from_cache": loaded_from_cache,
        "cache_path": str(cache_path) if cache_path is not None else None,
        "split_path": str(split_path),
        "split_method": SPLIT_ALIASES[str(split_method).lower()],
        "split_sizes": {name: int(indices.numel()) for name, indices in split_idx.items()},
        **summary,
    }
    return data_list, split_idx, metadata
