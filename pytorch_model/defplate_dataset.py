import torch
from torch.utils.data import Dataset
import torch
import time
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1
MESH_POS_INDEXES = slice(3, 6)  # like 3:6
RADIUS_WORLD_EDGE = 0.03

def add_edges(base_A, node_types, pos_t, radius):
    """
    Computes A_t dynamically using radius search.
    Excludes existing mesh edges (base_A) and self-loops.

    Returns:
        A_norm: Normalized adjacency matrix (including mesh edges + world edges + self loops)
        dynamic_edges: Edge list of ONLY the newly added world edges (2, E_world)
    """
    # Ensure devices match (likely CPU in Dataset)
    if base_A.device != pos_t.device:
        base_A = base_A.to(pos_t.device)

    # 1. Compute pairwise distances
    dists = torch.cdist(pos_t, pos_t)

    # 2. Radius mask (dist < r)
    radius_mask = dists < radius

    # 3. Exclude self-loops
    radius_mask.fill_diagonal_(False)

    # 4. Exclude existing mesh edges
    # base_A is already normalized, but non-zero entries imply edges
    mesh_edge_mask = base_A > 0

    # Valid world edges: in radius AND NOT in mesh
    valid_world_mask = radius_mask & (~mesh_edge_mask)

    # 5. Build combined adjacency
    # Convert base mesh to binary (1.0 for edge)
    binary_mesh = mesh_edge_mask.float()
    binary_world = valid_world_mask.float()

    A_combined = binary_mesh + binary_world

    # 6. Normalize
    row_sums = A_combined.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A_combined / row_sums

    # 7. Extract edge list for world edges (for return)
    # We want (source, target) pairs for world edges only
    dynamic_edges = torch.nonzero(binary_world, as_tuple=False).t()
    if dynamic_edges.numel() == 0:
         dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=pos_t.device)

    return A_norm, dynamic_edges


class DefPlateDataset(Dataset):
    total_comp_time = 0.0
    total_comp_calls = 0

    def __init__(self, list_of_trajs, add_world_edges=True):
        """
        Construct a dataset from a list of trajectories objects.

        :param list_of_trajs: List
            a list where each item has (A, X_seq_norm, mean, std, cells, node_type)
        :param add_world_edges: bool
            whether to add world edges dynamically based on radius search
        """
        self.samples = []
        # Store the list of trajectories
        self.trajs = list_of_trajs
        self.add_world_edges = add_world_edges

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
        """Returns the number of samples (trajectories)"""
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

        time_start = time.time()
        # 1. Get Features for t and t+1
        X_t = traj["X_seq_norm"][t]  # Shape: [N, F]
        X_tp1 = traj["X_seq_norm"][t + 1]  # Shape: [N, F]

        # 2. Get Static Data
        base_A = traj["A"]  # Shape: [N, N] (Static Mesh)
        node_types = traj["node_type"]  # Shape: [N]

        # 3. Compute Dynamic Adjacency
        # X_t layout: mesh_pos(0:3), world_pos(3:6), node_type(6:8), ...
        pos_t = X_t[:, MESH_POS_INDEXES]

        # This function adds edges between the ball and the plate based on proximity
        if self.add_world_edges:
            A_dynamic, dynamic_edges = add_edges(base_A=base_A, node_types=node_types, pos_t=pos_t, radius=RADIUS_WORLD_EDGE)
        else:
            A_dynamic = base_A
            dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=base_A.device)

        time_end = time.time()
        self.total_comp_time += time_start - time_end
        self.total_comp_calls += 1

        # Pass all inputs to model: mesh_pos, world_pos, node_type, vel, stress
        X_t_input = X_t

        return (A_dynamic, X_t_input, X_tp1, traj["mean"], traj["std"], traj["cells"], node_types, dynamic_edges,
            traj_id, t)


def collate_unet(batch):
    """
    Given a batch, we return a tuple of lists of the components of the tuple instead.

    :param batch: List
        a list of tuples (A, X_t, X_tp1, mean, std, cells, node_type, traj_id)

    :return (A_list, X_t_list, X_tp1_list, mean_list, std_list, cells_list, node_type_list, traj_id_list).
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

    for A, X_t, X_tp1, mean, std, cells, node_type, dyn_edges, traj_id, time_idx in batch:
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

    return (adjacency_mat_list, X_t_list, X_tp1_list, mean_list, std_list, cells_list,
        node_types_list, dynamic_edges_list, traj_id_list, time_idx_list)