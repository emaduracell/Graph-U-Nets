import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from torch_geometric.utils import coalesce, to_undirected
from tfrecord.torch.dataset import TFRecordDataset

# Feature layout (per node):
FEATURE_DIM = 12
TARGET_DIM = 4  # velocity (3) + stress (1)
NODE_TYPE_START = 3
NODE_TYPE_END = 5

STATS_FILE = "graph_unet_stats.pt"
INDEX_FILE = "graph_unet_index.pt"
TRAJ_TEMPLATE = "graph_unet_traj_{:05d}.pt"
PREPROCESSED_DIR = "graph_unet_preprocessed"


def _build_description(meta: Dict) -> Dict[str, str]:
    """Return TFRecord description mapping field names to raw byte loading."""
    return {k: "byte" for k in meta["field_names"]}


def _decode_record(rec_bytes: Dict[str, bytes], meta: Dict) -> Dict[str, np.ndarray]:
    """
    Decode a TFRecord example into numpy arrays following meta specs.
    """
    out = {}
    for k, v in rec_bytes.items():
        dtype = meta["features"][k]["dtype"]
        shape = meta["features"][k]["shape"]
        arr = np.frombuffer(v, dtype=getattr(np, dtype))
        arr = arr.reshape(shape)
        # Tile static features across time for convenience
        if meta["features"][k]["type"] == "static":
            arr = np.tile(arr, (meta["trajectory_length"], 1, 1))
        out[k] = arr
    return out


def _cell_to_edge_index(cells: np.ndarray) -> torch.Tensor:
    """Convert tetrahedral cells (C, 4) to undirected edge_index."""
    edge_indices = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    src = [cells[i][a] for i in range(cells.shape[0]) for a, _ in edge_indices]
    dst = [cells[i][b] for i in range(cells.shape[0]) for _, b in edge_indices]
    edges = torch.tensor([src, dst], dtype=torch.long)
    edges = to_undirected(edges)
    edges = coalesce(edges)
    if isinstance(edges, tuple):
        edges = edges[0]
    return edges


def _one_hot_node_type(node_type: np.ndarray) -> np.ndarray:
    """
    Map node_type {0: normal, 1: sphere, 3: boundary} -> 2-dim one-hot:
      - channel 0: normal vs others
      - channel 1: boundary
    Sphere nodes are treated as non-boundary (second channel 0).
    """
    lookup = np.array([
        [0, 0],  # 0 normal
        [1, 0],  # 1 sphere
        [0, 0],  # 2 unused
        [0, 1],  # 3 boundary
    ], dtype=np.float32)
    node_type = node_type.squeeze(-1)  # (N,)
    return lookup[node_type]


def _normalize_id_list(ids):
    """Accept int, list of int, or nested lists and return flat list of ints or None."""
    if ids is None:
        return None
    if isinstance(ids, int):
        return [int(ids)]
    if isinstance(ids, (list, tuple)):
        flat = []
        for v in ids:
            if isinstance(v, (list, tuple)):
                flat.extend([int(x) for x in v])
            else:
                flat.append(int(v))
        return flat
    # Fallback: try to cast
    return [int(ids)]
class GraphUNetTFRecordDataset(Dataset):
    """
    Lazy dataset backed by preprocessed trajectory files.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        preprocessed_dir: Optional[str] = None,
        allowed_traj_ids: Optional[List[int]] = None,
        allowed_time_ids: Optional[List[int]] = None,
        transform=None,  # <--- NEW ARGUMENT ADDED HERE
    ):
        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.pre_dir = preprocessed_dir or os.path.join(data_dir, PREPROCESSED_DIR)
        self.transform = transform # <--- STORE THE TRANSFORM

        # Load stats and index
        stats_path = os.path.join(self.pre_dir, STATS_FILE)
        index_path = os.path.join(self.pre_dir, INDEX_FILE)
        if not os.path.exists(stats_path) or not os.path.exists(index_path):
            raise FileNotFoundError(
                f"Preprocessed data not found in {self.pre_dir}. "
                f"Please run pyg_preprocess.py first."
            )

        stats = torch.load(stats_path, map_location="cpu")
        self.mean_feat = stats["mean_feat"].float()
        self.std_feat = stats["std_feat"].float()
        self.mean_target = stats["mean_target"].float()
        self.std_target = stats["std_target"].float()
        self.feature_dim = FEATURE_DIM
        self.target_dim = TARGET_DIM

        index_data = torch.load(index_path, map_location="cpu")
        index_list: List[Tuple[int, int]] = index_data["index"]

        # Optional filtering
        allowed_traj_ids = _normalize_id_list(allowed_traj_ids)
        allowed_time_ids = _normalize_id_list(allowed_time_ids)
        allowed_traj_set = set(allowed_traj_ids) if allowed_traj_ids else None
        allowed_time_set = set(allowed_time_ids) if allowed_time_ids else None
        if allowed_traj_set or allowed_time_set:
            index_list = [
                (tr, t)
                for (tr, t) in index_list
                if (allowed_traj_set is None or tr in allowed_traj_set)
                and (allowed_time_set is None or t in allowed_time_set)
            ]

        self.index_list: List[Tuple[int, int]] = index_list
        self.available_traj = set(idx for idx, _ in self.index_list)
        self._traj_cache = {"idx": None, "data": None}

    def __len__(self) -> int:
        return len(self.index_list)

    def _load_traj(self, traj_idx: int):
        if self._traj_cache["idx"] == traj_idx:
            return self._traj_cache["data"]
        traj_path = os.path.join(self.pre_dir, TRAJ_TEMPLATE.format(traj_idx))
        if not os.path.exists(traj_path):
            raise FileNotFoundError(f"Trajectory file missing: {traj_path}")
        data = torch.load(traj_path, map_location="cpu")
        self._traj_cache = {"idx": traj_idx, "data": data}
        return data

    def __getitem__(self, idx: int) -> Data:
        traj_idx, time_idx = self.index_list[idx]
        traj = self._load_traj(traj_idx)
        
        # Load tensors
        x = traj["x_seq"][time_idx].float()
        y = traj["y_seq"][time_idx].float()
        node_type = traj["node_type"].long()
        edge_index = traj["edge_index"].long()
        
        # Create Data object
        data = Data(
            x=x,
            y=y,
            edge_index=edge_index,
            node_type=node_type,
        )

        # <--- APPLY TRANSFORM HERE
        if self.transform is not None:
            data = self.transform(data)
            
        return data