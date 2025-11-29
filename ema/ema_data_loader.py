import json
import numpy as np
import torch
from tfrecord.reader import tfrecord_loader
# ============================================================

def decode_trajectory_from_record(record, meta):
    """
    Identical logic to your EGNN loader.
    Only decodes the TFRecord into numpy arrays.
    """
    def normalize_to_bytes(value):
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return bytes(value.flat[0])
            raise TypeError("normalize_to_bytes chiamato su ndarray numerico")
        raise TypeError(f"Tipo inatteso: {type(value)}")

    def decode_raw_array(value, dtype, shape_spec):
        if isinstance(value, np.ndarray) and value.dtype != object:
            arr = value.astype(dtype)
            tgt = list(shape_spec)
            if -1 in tgt:
                known = np.prod([d for d in tgt if d != -1])
                missing = arr.size // known
                tgt[tgt.index(-1)] = missing
            if tuple(arr.shape) != tuple(tgt):
                arr = arr.reshape(tgt)
            return arr

        raw = normalize_to_bytes(value)
        arr = np.frombuffer(raw, dtype=dtype)

        tgt = list(shape_spec)
        if -1 in tgt:
            known = np.prod([d for d in tgt if d != -1])
            missing = arr.size // known
            tgt[tgt.index(-1)] = missing

        return arr.reshape(tgt)

    shapes = meta["features"]

    world_pos = decode_raw_array(record["world_pos"], np.float32,
                                 shapes["world_pos"]["shape"])
    stress = decode_raw_array(record["stress"], np.float32,
                              shapes["stress"]["shape"])
    node_type = decode_raw_array(record["node_type"], np.int32,
                                 shapes["node_type"]["shape"])
    mesh_pos = decode_raw_array(record["mesh_pos"], np.float32,
                                shapes["mesh_pos"]["shape"])
    cells = decode_raw_array(record["cells"], np.int32,
                             shapes["cells"]["shape"])

    # remove extra dims
    if node_type.shape[0] == 1:
        node_type = node_type[0]
    if mesh_pos.shape[0] == 1:
        mesh_pos = mesh_pos[0]
    if cells.shape[0] == 1:
        cells = cells[0]

    return {
        "world_pos": world_pos.astype(np.float32),  # (T,N,3)
        "stress": stress.astype(np.float32),        # (T,N,1)
        "node_type": node_type.astype(np.int32),    # (N,1)
        "mesh_pos": mesh_pos.astype(np.float32),    # (N,3)
        "cells": cells.astype(np.int32),            # (C,4)
    }


def load_raw_trajectory_from_tfrecord(tfrecord_path, meta, traj_index):
    loader = tfrecord_loader(tfrecord_path, index_path=None)
    for i, record in enumerate(loader):
        if i == traj_index:
            return decode_trajectory_from_record(record, meta)
    raise IndexError(f"trajectory index {traj_index} out of range")

# ============================================================

def adjacency_from_cells(cells, num_nodes):
    """
    Build binary undirected adjacency matrix A ∈ R^{N×N}.
    Option A: NO self loops.
    """
    A = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)

    for c in cells:
        i0, i1, i2, i3 = map(int, c.tolist())
        quad = [i0, i1, i2, i3]
        # Each cell forms 4 edges in a cycle
        for u, v in zip(quad, quad[1:] + quad[:1]):
            if u != v:
                A[u, v] = 1.0
                A[v, u] = 1.0  # undirected
    return A
# ============================================================

def trajectory_to_gnet_inputs(traj):
    """
    Create:
    - adjacency matrix A    [N,N]
    - feature sequence X_seq [T,N,F_in]
    where F_in = 3 (coords) + node_type + velocity + stress
    """

    world_pos = traj["world_pos"]   # (T,N,3)
    stress    = traj["stress"]      # (T,N,1)
    node_type = traj["node_type"]   # (N,1)
    cells     = traj["cells"]       # (C,4)

    T, N, _ = world_pos.shape

    # ----- Build velocities (same as EGNN) -----
    vel = np.zeros((T, N, 3), dtype=np.float32)
    if T > 1:
        vel[1] = world_pos[1] - world_pos[0]
    for t in range(2, T):
        vel[t] = world_pos[t] - world_pos[t - 1]

    # ----- Node type → float for concatenation -----
    node_type_f = node_type.astype(np.float32)  # (N,1)

    # ----- Build X_seq -----
    X_list = []
    for t in range(T):
        feats_t = np.concatenate(
            [
                world_pos[t],    # (N,3)
                node_type_f,     # (N,1)
                vel[t],          # (N,3)
                stress[t],       # (N,1)
            ],
            axis=-1
        )  # (N, 3+1+3+1 = 8)
        X_list.append(feats_t)

    X_seq = torch.tensor(np.stack(X_list, axis=0), dtype=torch.float32)  # [T,N,F_in]

    # ----- Build adjacency -----
    A = adjacency_from_cells(cells, num_nodes=N)  # [N,N]
    mean = X_seq.mean(dim=(0,1), keepdim=True)         # [1,1,F]
    std  = X_seq.std(dim=(0,1), keepdim=True) + 1e-6   # [1,1,F]

    X_seq_norm = (X_seq - mean) / std
    return A, X_seq_norm, mean, std, traj["cells"]
# ============================================================

def data_loader_gnet(tfrecord_path, meta_path, traj_index):
    """
    Main entrypoint:
    loads trajectory, builds adjacency and feature seq.
    """
    with open(meta_path, "r") as f:
        meta = json.load(f)

    traj_dict = load_raw_trajectory_from_tfrecord(
        tfrecord_path, meta, traj_index
    )

    return trajectory_to_gnet_inputs(traj_dict)
