import torch
from torch.utils.data import Dataset

def create_dynamic_adjacency_on_the_fly(base_A, node_types, pos_t, k=2):
    """
    Computes A_t dynamically.
    For each Sphere node, it looks at its k-nearest neighbors in the *entire* graph.
    If a neighbor is a Plate node (Normal or Boundary), it adds an edge.
    If a neighbor is another Sphere node, it ignores it (assumed already handled or irrelevant).
    """
    A_t = base_A.clone() 
    
    # 1. Identify Sphere indices
    sphere_indices = torch.nonzero(node_types == 1, as_tuple=True)[0]
    
    # We need to look at ALL nodes for neighbors, so we use pos_t for everyone.
    # To save time, we compute distance from Sphere Nodes -> All Nodes.
    
    if len(sphere_indices) > 0:
        sphere_pos = pos_t[sphere_indices]
        all_pos = pos_t
        
        # Compute distances: [N_sphere, N_total]
        dists = torch.cdist(sphere_pos, all_pos)
        
        # Find k+1 nearest neighbors (including itself, so we take k+1 and skip index 0 later)
        # actually, just taking k is fine if we check identity, but let's take k+1 to be safe
        k_val = min(k + 1, len(all_pos))
        _, neighbor_indices = torch.topk(dists, k=k_val, dim=1, largest=False)
        
        # neighbor_indices is [N_sphere, k+1] containing global indices of neighbors
        
        # Iterate over sphere nodes and their found neighbors
        for i, sphere_global_idx in enumerate(sphere_indices):
            # Get the global indices of the k nearest neighbors for this sphere node
            global_neighbors = neighbor_indices[i]
            
            for nb_global_idx in global_neighbors:
                # 1. Skip self-loops (if dist was 0)
                if nb_global_idx == sphere_global_idx:
                    continue
                
                # 2. Check the TYPE of the neighbor
                nb_type = node_types[nb_global_idx]
                
                # If neighbor is Plate (0) or Boundary (3), add connection
                if nb_type == 0 or nb_type == 3:
                    A_t[sphere_global_idx, nb_global_idx] = 1.0
                    A_t[nb_global_idx, sphere_global_idx] = 1.0
                
                # If neighbor is Sphere (1), we do nothing (don't add extra edges between sphere nodes)

    # Normalize
    row_sums = A_t.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A_t / row_sums
    
    return A_norm

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
        A_dynamic = create_dynamic_adjacency_on_the_fly(
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
    cells_list = []
    node_types_list = []
    traj_id_list = []
    time_idx_list = []

    for A, X_t, X_tp1, mean, std, cells, node_type, traj_id, time_idx in batch:
        adjacency_mat_list.append(A)
        X_t_list.append(X_t)
        X_tp1_list.append(X_tp1)
        mean_list.append(mean)
        std_list.append(std)
        cells_list.append(cells)
        node_types_list.append(node_type)
        traj_id_list.append(traj_id)
        time_idx_list.append(time_idx)

    return (
        adjacency_mat_list, 
        X_t_list, 
        X_tp1_list, 
        mean_list, 
        std_list, 
        cells_list, 
        node_types_list, 
        traj_id_list,
        time_idx_list
    )