import torch
from torch.utils.data import Dataset
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1


class DefPlateDataset(Dataset):
    total_comp_time = 0.0
    total_comp_calls = 0

    def __init__(self, list_of_trajs, world_pos_idxs, velocity_idxs):
        """
        Construct a dataset from a list of trajectories objects.

        :param list_of_trajs: List
            a list where each item has (A, X_seq_norm, mean, std, cells, node_type)
        """
        self.samples = []
        # Store the list of trajectories
        self.trajs = list_of_trajs
        self.world_pos_idxs = world_pos_idxs
        self.velocity_idxs = velocity_idxs

        for traj_id, traj in enumerate(list_of_trajs):
            X_seq = traj["X_seq_norm"]
            T = X_seq.shape[0]

            # Indexing: We create a sample for every transition t -> t+1
            for t in range(T - 1):
                self.samples.append({"traj_id": traj_id, "time_idx": t})

    def __len__(self):
        """Returns the number of samples (trajectories)"""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches the sample and computes the dynamic adjacency matrix.
        """
        # Retrieve trajectory and its features
        s = self.samples[idx]
        traj_id = s["traj_id"]
        t = s["time_idx"]
        traj = self.trajs[traj_id]
        X_t_input = traj["X_seq_norm"][t]
        X_tp1_target = traj["X_seq_norm"][t + 1]
        
        # Check if A is dynamic [T, N, N] or static [N, N] (backward compatibility)
        if traj["A"].ndim == 3:
            base_A = traj["A"][t]
        else:
            base_A = traj["A"]
            
        node_types = traj["node_type"]

        return base_A, X_t_input, X_tp1_target, traj["mean"], traj["std"], traj["cells"], node_types, traj_id, t

def collate_unet(batch):
    """
    Given a batch, we return a tuple of lists of the components of the tuple instead.

    Args:
        batch: List
            A list of tuples (A, X_t, X_tp1, mean, std, cells, node_type, traj_id)

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

    return (adjacency_mat_list, X_t_list, X_tp1_list, mean_list, std_list, cells_list,
        node_types_list, traj_id_list, time_idx_list)
    # base_A, X_t_input, X_tp1_target, traj["mean"], traj["std"], traj["cells"], node_types, traj_id, t


def collate_block_diagonal(batch):
    """
    Collate samples into a single disjoint-union graph:
      - block-diagonal adjacency (Sparse COO)
      - concatenated feature and node-type tensors
    This is optimized for vectorized training on large GPUs.
    """
    x_t_list = []
    x_tp1_list = []
    node_types_list = []

    all_indices = []
    all_values = []
    cumulative_nodes = 0

    for A, X_t, X_tp1, _, _, _, node_type, _, _ in batch:
        # A is likely dense [N, N] here from process_single_trajectory
        # Convert to sparse COO indices/values
        if A.is_sparse:
            print(f"[collate_block_diagonal] A is sparse, shape: {A.shape}")
             A_coo = A.coo()
             indices = A_coo.indices()
             values = A_coo.values()
        else:
            print(f"[collate_block_diagonal] A is not sparse, shape: {A.shape}")
             indices = torch.nonzero(A).t()
             values = A[indices[0], indices[1]]

        # Shift indices
        shifted_indices = indices + cumulative_nodes
        all_indices.append(shifted_indices)
        all_values.append(values)

        x_t_list.append(X_t)
        x_tp1_list.append(X_tp1)
        node_types_list.append(node_type)
        
        cumulative_nodes += X_t.shape[0]

    # Create sparse block-diagonal adjacency
    if len(all_indices) > 0:
        batch_indices = torch.cat(all_indices, dim=1)
        batch_values = torch.cat(all_values)
    else:
        batch_indices = torch.empty((2, 0), dtype=torch.long)
        batch_values = torch.empty(0)

    batch_adj = torch.sparse_coo_tensor(
        batch_indices, 
        batch_values, 
        (cumulative_nodes, cumulative_nodes)
    ).coalesce()

    batch_x_t = torch.cat(x_t_list, dim=0)
    batch_x_tp1 = torch.cat(x_tp1_list, dim=0)
    batch_node_types = torch.cat(node_types_list, dim=0)

    return batch_adj, batch_x_t, batch_x_tp1, batch_node_types