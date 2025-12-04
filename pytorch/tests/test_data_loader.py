import sys
import os
# sys.path.append("")
from data_loader import load_all_trajectories
import torch
import numpy as np

def adj_sanity(A, mesh_cells):
    # symmetry check
    sym_diff = (A - A.T).abs().sum()
    print("Symmetry error:", sym_diff.item())  # should be 0 before normalization

    # consistency with mesh_cells for a small example
    for c in mesh_cells[:5]:  # just first few cells
        i0, i1, i2, i3 = map(int, c.tolist())
        quad = [i0, i1, i2, i3]
        for u, v in zip(quad, quad[1:] + quad[:1]):
            assert A[u, v] > 0, f"Missing edge {u}->{v}"
            assert A[v, u] > 0, f"Missing edge {v}->{u}"

def dataset_sanity_checks(list_of_trajs):
    """
    Run a bunch of sanity checks on the loaded dataset.
    """

    # --- 1) Check adjacency rows approximately sum to 1 ---
    for i, traj in enumerate(list_of_trajs):
        A = traj["A"]
        row_sums = A.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            diff_tensor = (row_sums - 1.0).abs()
            diff = diff_tensor.max().item()
            bad_rows = torch.nonzero(diff_tensor > 1e-5, as_tuple=False).view(-1)

            print(f"\n[dataset_sanity_checks] DEBUG trajectory {i}")
            print("  row_sums[bad_rows][:10] =", row_sums[bad_rows][:10])
            print("  number of bad rows:", bad_rows.numel())
            print("  min row sum:", row_sums.min().item())
            print("  max row sum:", row_sums.max().item())
            print(f"\tA bad_rows =\n {A[bad_rows]}")

            raise ValueError(f"[A] Row sums for trajectory {i} deviate from 1 by up to {diff}")
    print("[A] ✔ all row sums are ≈ 1 (row-normalized).")

    # --- 2) Check features for NaNs / infs ---
    for i, traj in enumerate(list_of_trajs):
        X = traj["X_seq_norm"]  # [T, N, F]
        if torch.isnan(X).any():
            raise ValueError(f"[X_seq_norm] NaNs found in trajectory {i}")
        if torch.isinf(X).any():
            raise ValueError(f"[X_seq_norm] Infs found in trajectory {i}")
    print("[X_seq_norm] ✔ no NaNs or Infs found.")

    # --- 3) Inspect node_type values across the whole dataset ---
    all_node_types = torch.cat(
        [traj["node_type"].view(-1) for traj in list_of_trajs], dim=0
    )
    unique_types = torch.unique(all_node_types).tolist()
    print(f"[node_type] unique values across dataset: {unique_types}")


def inspect_traj_adjacency(traj, idx):
    A = traj["A"]
    cells = traj["cells"]
    node_type = traj["node_type"]
    N = A.shape[0]

    row_sums = A.sum(dim=1)
    bad_rows = torch.nonzero((row_sums - 1.0).abs() > 1e-5, as_tuple=False).view(-1)

    print(f"\n[inspect_traj_adjacency] trajectory {idx}")
    print("  N =", N)
    print("  bad_rows indices:", bad_rows.tolist())
    print("  row_sums[bad_rows] =", row_sums[bad_rows].tolist())
    print("  max index in cells:", cells.max().item() if cells.numel() > 0 else "no cells")
    print("  node_type shape:", node_type.shape)
    print("  A[bad_rows]:\n", A[bad_rows])

if __name__ == "__main__":
    TFRECORD_PATH = "data/train.tfrecord"
    META_PATH = "data/meta.json"
    NUM_TRAIN_TRAJS = 1500  # Load only the first K trajectories

    list_of_trajs = load_all_trajectories(TFRECORD_PATH, META_PATH, NUM_TRAIN_TRAJS)

    for idx, traj in enumerate(list_of_trajs):
        A = traj['A']
        row_sums = A.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            print(f"[load_all_trajectories] BAD A just built for traj {idx}")
            print("  min row sum:", row_sums.min().item())
            print("  max row sum:", row_sums.max().item())
            inspect_traj_adjacency(traj, idx)

    # Run dataset-wide sanity checks
    # dataset_sanity_checks(list_of_trajs)