import json
import torch
import numpy as np
from tfrecord.reader import tfrecord_loader

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

#  LOAD MULTIPLE TRAJECTORIES (MEMORY-SAFE, WITH max_trajs OPTION)

def build_edges_from_cells(cells, num_nodes):
    edge_set = set()

    for c in cells:
        i0, i1, i2, i3 = map(int, c.tolist())
        quad = [i0, i1, i2, i3]

        for u, v in zip(quad, quad[1:] + quad[:1]):
            if u != v:
                edge_set.add((u, v))
                edge_set.add((v, u))

    edge_list = sorted(edge_set)
    return torch.tensor(edge_list, dtype=torch.long)

def load_all_trajectories(tfrecord_path, meta_path, max_trajs=None):
    """
    Load up to `max_trajs` trajectories from TFRecord.

    Returns
    -------
    list_of_trajs : list of dicts
        Each dict contains:
          - "A"          : [N,N] adjacency matrix (torch.float32)
          - "X_seq_norm" : [T,N,F] normalized features (torch.float32)
          - "mean"       : [1,1,F] mean for denorm
          - "std"        : [1,1,F] std for denorm
          - "cells"      : [C,4] connectivity (torch.long)
    """

    # LOAD META.JSON (NEEDED FOR DECODING)
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # TFRecord loader
    loader = tfrecord_loader(tfrecord_path, index_path=None)

    list_of_trajs = []


    # Iterate through trajectories
    for traj_idx, record in enumerate(loader):

        # Stop if we reached max_trajs
        if max_trajs is not None and traj_idx >= max_trajs:
            break

        # DECODE RAW TRAJECTORY 
        traj = decode_trajectory_from_record(record, meta)
        world_pos = traj["world_pos"]      # (T,N,3)
        stress    = traj["stress"]         # (T,N,1)
        node_type = traj["node_type"]      # (N,1)
        cells     = traj["cells"]          # (C,4)

        T, N, _ = world_pos.shape

        # BUILD VELOCITY 
        vel = np.zeros((T, N, 3), dtype=np.float32)
        for t in range(1, T):
            vel[t] = world_pos[t] - world_pos[t-1]

        # BUILD FEATURE SEQUENCE 
        # Feature layout:
        #   [pos_x, pos_y, pos_z, node_type, vel_x, vel_y, vel_z, stress]
        feats_list = []
        node_type_f = node_type.astype(np.float32)

        for t in range(T):
            feats_t = np.concatenate(
                [
                    world_pos[t],      # (N,3)
                    node_type_f,       # (N,1)
                    vel[t],            # (N,3)
                    stress[t],         # (N,1)
                ],
                axis=-1
            )
            feats_list.append(feats_t)

        X_seq = torch.tensor(np.stack(feats_list, axis=0), dtype=torch.float32)   # [T,N,F]

        # NORMALIZATION PER TRAJECTORY 
        mean = X_seq.mean(dim=(0, 1), keepdim=True)
        std  = X_seq.std(dim=(0, 1), keepdim=True) + 1e-6
        X_seq_norm = (X_seq - mean) / std

        # ADJACENCY MATRIX 
        edge_index = build_edges_from_cells(cells, num_nodes=N)

        A = torch.zeros((N, N), dtype=torch.float32)
        for e in edge_index:
            A[e[0], e[1]] = 1.0

        # add self-loops
        A = A + torch.eye(N)

        # row-normalize
        A = A / A.sum(dim=1, keepdim=True)

        # ensure cells is tensor
        cells_t = torch.tensor(cells, dtype=torch.long)

        # STORE TRAJECTORY 
        list_of_trajs.append({
            "A": A,
            "X_seq_norm": X_seq_norm,
            "mean": mean,
            "std": std,
            "cells": cells_t
        })

        print(f"Loaded trajectory {traj_idx}, shape = {X_seq_norm.shape}")

    # RETURN
    print(f"\nLoaded {len(list_of_trajs)} trajectories.")
    return list_of_trajs
