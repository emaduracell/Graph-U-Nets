import torch
from torch.utils.data import Dataset


class EmaUnetDataset(Dataset):

    def __init__(self, list_of_trajs):
        """
        Construct an instance.

        :param list_of_trajs: List
            a list where each item has (A, X_seq_norm, mean, std, cells, node_type)
        """
        self.samples = []

        for traj_id, traj in enumerate(list_of_trajs):
            A = traj["A"]
            X_seq = traj["X_seq_norm"]
            mean = traj["mean"]
            std = traj["std"]
            cells = traj["cells"]
            node_type = traj["node_type"]

            T = X_seq.shape[0]

            for t in range(T - 1):
                self.samples.append({
                    "A": A,
                    "X_t": X_seq[t],
                    "X_tp1": X_seq[t+1],
                    "mean": mean,
                    "std": std,
                    "cells": cells,
                    "node_type": node_type,
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
            s["node_type"], # [N]
            s["traj_id"]
        )

def collate_ema_unet(batch):
    """
    Given a batch, we return a tuple of lists of the components of the tuple instead

    :param batch: List
        a list of tuples (A, X_t, X_tp1, mean, std, cells, node_type, traj_id)
    :return (A_list, X_t_list, X_tp1_list, mean_list, std_list, cells_list, node_type_list, traj_id_list).
    """

    adjacency_mat_list = []
    X_t_list = []
    X_tp1_list = []
    mean_list = []
    std_list = []
    cells_list = []
    node_types_list = []
    traj_id_list = []

    for A, X_t, X_tp1, mean, std, cells, node_type, traj_id in batch:
        adjacency_mat_list.append(A)
        X_t_list.append(X_t)
        X_tp1_list.append(X_tp1)
        mean_list.append(mean)
        std_list.append(std)
        cells_list.append(cells)
        node_types_list.append(node_type)
        traj_id_list.append(traj_id)

    return adjacency_mat_list, X_t_list, X_tp1_list, mean_list, std_list, cells_list, node_types_list, traj_id_list
