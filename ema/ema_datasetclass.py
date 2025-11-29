import torch
from torch.utils.data import Dataset


class EmaUnetDataset(Dataset):

    def __init__(self, list_of_trajs):
        """
        list_of_trajs is a list where each item has:
            A, X_seq_norm [T,N,F], mean, std, cells
        """
        self.samples = []

        for traj_id, traj in enumerate(list_of_trajs):
            A      = traj["A"]
            X_seq  = traj["X_seq_norm"]
            mean   = traj["mean"]
            std    = traj["std"]
            cells  = traj["cells"]

            T = X_seq.shape[0]

            for t in range(T - 1):
                self.samples.append({
                    "A": A,
                    "X_t": X_seq[t],
                    "X_tp1": X_seq[t+1],
                    "mean": mean,
                    "std": std,
                    "cells": cells,
                    "traj_id": traj_id
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            s["A"],         # [N,N]
            s["X_t"],       # [N,F]
            s["X_tp1"],     # [N,F]
            s["mean"],      # [1,1,F]
            s["std"],       # [1,1,F]
            s["cells"],     # [C,4]
            s["traj_id"]
        )
def collate_ema_unet(batch):
    """
    batch is a list of:
      (A, X_t, X_tp1, mean, std, cells, traj_id)

    We return lists, not tensors, because graphs have different sizes.
    """

    As = []
    X_ts = []
    X_tp1s = []
    means = []
    stds = []
    cells_list = []
    traj_ids = []

    for A, X_t, X_tp1, mean, std, cells, traj_id in batch:
        As.append(A)
        X_ts.append(X_t)
        X_tp1s.append(X_tp1)
        means.append(mean)
        stds.append(std)
        cells_list.append(cells)
        traj_ids.append(traj_id)

    return As, X_ts, X_tp1s, means, stds, cells_list, traj_ids
