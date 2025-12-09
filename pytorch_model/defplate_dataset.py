import torch
from torch.utils.data import Dataset
import torch

def create_dynamic_adjacency_on_the_fly(base_A, node_types, pos_t, k=25):
    """
        Computes A_t dynamically.
    For each Sphere node, it looks at its k-nearest neighbors in the *entire* graph.
    If a neighbor is a Plate node (Normal or Boundary), it adds an edge.
    If a neighbor is another Sphere node, it ignores it (assumed already handled or irrelevant).
    Vectorized version: No Python loops.
    """
    A_t = base_A.clone()
    
    # 1. Identify Sphere indices
    sphere_indices = torch.nonzero(node_types == 1, as_tuple=True)[0]
    
    if len(sphere_indices) > 0:
        sphere_pos = pos_t[sphere_indices]
        
        # 2. Compute Distances & TopK (Vectorized)
        # Shape: [N_sphere, N_total]
        dists = torch.cdist(sphere_pos, pos_t)
        
        # Get k+1 neighbors (to safely account for self-loops)
        k_val = min(k + 1, len(pos_t))
        _, neighbor_indices = torch.topk(dists, k=k_val, dim=1, largest=False)
        # neighbor_indices shape: [N_sphere, k_val]
        
        # 3. Filter Valid Neighbors (Vectorized)
        
        # Get types of all neighbors at once
        # Shape: [N_sphere, k_val]
        nb_types = node_types[neighbor_indices] 
        
        # Create a boolean mask for valid connections:
        # Condition A: Neighbor must be Plate (0) or Boundary (3)
        type_mask = (nb_types == 0) | (nb_types == 3)
        
        # Condition B: Neighbor must not be self 
        # (Compare neighbor index to sphere index broadcasted)
        # sphere_indices shape: [N_sphere] -> [N_sphere, 1]
        self_mask = neighbor_indices != sphere_indices.unsqueeze(1)
        
        # Combine masks
        valid_mask = type_mask & self_mask
        
        # 4. Apply updates to A_t
        # Extract source indices (spheres) and target indices (neighbors) where mask is True
        source_idxs = sphere_indices.unsqueeze(1).expand_as(neighbor_indices)[valid_mask]
        target_idxs = neighbor_indices[valid_mask]
        
        # Batch update the adjacency matrix
        # Note: We use indices directly rather than looping
        A_t.index_put_((source_idxs, target_idxs), torch.tensor(1.0, device=A_t.device))
        A_t.index_put_((target_idxs, source_idxs), torch.tensor(1.0, device=A_t.device))
        
        # Create edge list for return
        if len(source_idxs) > 0:
            dynamic_edges = torch.stack([source_idxs, target_idxs], dim=0)
        else:
            dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=A_t.device)

    else:
        dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=A_t.device)

    # 5. Normalize (Vectorized)
    row_sums = A_t.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A_t / row_sums
    
    return A_norm, dynamic_edges


class DefPlateDataset(Dataset):
    def __init__(self, list_of_trajs):
        """
        Construct a dataset from a list of trajectories objects.
        
        :param list_of_trajs: List
            a list where each item is a dictionary containing:
            "A", "X_seq_norm", "mean", "std", "cells", "node_type"
        """
        self.samples = []
        # Store the list of trajectories
        self.trajs = list_of_trajs

        for traj_id, traj in enumerate(list_of_trajs):
            X_seq = traj["X_seq_norm"]
            T = X_seq.shape[0]

            # Indexing: We create a sample for every transition t -> t+1
            for t in range(T - 1):
                self.samples.append({
                    "traj_id": traj_id,
                    "time_idx": t,
                })

    def __len__(self):
        """Returns the total number of samples (time steps across all trajs)"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches the sample and computes the dynamic adjacency matrix on-the-fly.
        """
        s = self.samples[idx]
        traj_id = s["traj_id"]
        t = s["time_idx"]
        
        # Retrieve the trajectory
        traj = self.trajs[traj_id]
        
        # 1. Get Features for t and t+1
        X_t = traj["X_seq_norm"][t]       # Shape: [N, F]
        X_tp1 = traj["X_seq_norm"][t+1]   # Shape: [N, F]
        
        # 2. Get Static Data
        base_A = traj["A"]                # Shape: [N, N] (Static Mesh)
        node_types = traj["node_type"]    # Shape: [N]
        
        # 3. Compute Dynamic Adjacency
        # We assume the first 3 columns of X_t are positions (x, y, z)
        pos_t = X_t[:, :3]
        
        # This function adds edges between the ball and the plate based on proximity
        A_dynamic, dynamic_edges = create_dynamic_adjacency_on_the_fly(
            base_A=base_A, 
            node_types=node_types, 
            pos_t=pos_t
        )

        return (
            A_dynamic,
            X_t,
            X_tp1,
            traj["mean"],
            traj["std"],
            traj["cells"],
            node_types,
            dynamic_edges,
            traj_id,
            t
        )

def collate_unet(batch):
    """
    Custom collate function to batch the data as lists of tensors.
    """
    adjacency_mat_list = []
    X_t_list = []
    X_tp1_list = []
    mean_list = []
    std_list = []
    dynamic_edges_list = []
    cells_list = []
    node_types_list = []
    traj_id_list = []
    time_idx_list = []

    for A, X_t, X_tp1, mean, std, cells, node_type,dyn_edges, traj_id, time_idx in batch:
        adjacency_mat_list.append(A)
        X_t_list.append(X_t)
        X_tp1_list.append(X_tp1)
        mean_list.append(mean)
        std_list.append(std)
        cells_list.append(cells)
        node_types_list.append(node_type)
        traj_id_list.append(traj_id)
        time_idx_list.append(time_idx)
        dynamic_edges_list.append(dyn_edges)

    return (
        adjacency_mat_list, 
        X_t_list, 
        X_tp1_list, 
        mean_list, 
        std_list, 
        cells_list, 
        node_types_list, 
        dynamic_edges_list,
        traj_id_list,
        time_idx_list
    )