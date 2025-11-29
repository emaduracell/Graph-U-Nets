import json
import torch
import numpy as np
from tfrecord.reader import tfrecord_loader
from ema_data_loader import decode_trajectory_from_record 
from data_loader_egnn import build_edges_from_cells


# =====================================================================
#  LOAD MULTIPLE TRAJECTORIES (MEMORY-SAFE, WITH max_trajs OPTION)
# =====================================================================

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

    # -------------------------------------
    # Load meta.json (needed for decoding)
    # -------------------------------------
    with open(meta_path, "r") as f:
        meta = json.load(f)

    # -------------------------------------
    # TFRecord loader
    # -------------------------------------
    loader = tfrecord_loader(tfrecord_path, index_path=None)

    list_of_trajs = []

    # -------------------------------------
    # Iterate through trajectories
    # -------------------------------------
    for traj_idx, record in enumerate(loader):

        # Stop if we reached max_trajs
        if max_trajs is not None and traj_idx >= max_trajs:
            break

        # -------------------------- DECODE RAW TRAJECTORY ---------------------------
        traj = decode_trajectory_from_record(record, meta)
        world_pos = traj["world_pos"]      # (T,N,3)
        stress    = traj["stress"]         # (T,N,1)
        node_type = traj["node_type"]      # (N,1)
        cells     = traj["cells"]          # (C,4)

        T, N, _ = world_pos.shape

        # -------------------------- BUILD VELOCITY ---------------------------
        vel = np.zeros((T, N, 3), dtype=np.float32)
        for t in range(1, T):
            vel[t] = world_pos[t] - world_pos[t-1]

        # -------------------------- BUILD FEATURE SEQUENCE -------------------------
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

        # -------------------------- NORMALIZATION PER TRAJECTORY -------------------------
        mean = X_seq.mean(dim=(0, 1), keepdim=True)
        std  = X_seq.std(dim=(0, 1), keepdim=True) + 1e-6
        X_seq_norm = (X_seq - mean) / std

        # -------------------------- ADJACENCY MATRIX -------------------------
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

        # -------------------------- STORE TRAJECTORY -------------------------
        list_of_trajs.append({
            "A": A,
            "X_seq_norm": X_seq_norm,
            "mean": mean,
            "std": std,
            "cells": cells_t
        })

        print(f"Loaded trajectory {traj_idx}, shape = {X_seq_norm.shape}")

    # ----------------------------------------------------------------------
    # RETURN
    # ----------------------------------------------------------------------
    print(f"\nLoaded {len(list_of_trajs)} trajectories.")
    return list_of_trajs
