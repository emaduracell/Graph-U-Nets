import torch
from torch.utils.data import Dataset
import torch
import time
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1

def add_w_edges_neigh(base_A, node_types, pos_t, k):
    """
    For each Sphere node, it looks at its k-nearest neighbors.
    If a neighbor is a Plate node (Normal or Boundary), it adds an edge.
    If a neighbor is another Sphere node, it ignores it (assumed already handled or irrelevant).

    :param base_A
    :param node_types
    :param pos_t
    :param radius

    :return: (A_norm, dynamic_edges)
    """
    A_t = base_A.clone()

    # 1. Identify Sphere indices
    sphere_indices = torch.nonzero(node_types == SPHERE_NODE, as_tuple=True)[0]

    if len(sphere_indices) > 0:

        # Start from sphere indices and get
        sphere_pos = pos_t[sphere_indices]
        dists = torch.cdist(sphere_pos, pos_t)
        k_val = min(k + 1, len(pos_t))
        _, neighbor_indices = torch.topk(dists, k=k_val, dim=1, largest=False)

        # Get type of neighbors and create a boolean mask for valid connections, then apply
        nb_types = node_types[neighbor_indices]
        type_mask = (nb_types == NORMAL_NODE) | (nb_types == BOUNDARY_NODE)
        self_mask = neighbor_indices != sphere_indices.unsqueeze(1)
        valid_mask = type_mask & self_mask
        source_idxs = sphere_indices.unsqueeze(1).expand_as(neighbor_indices)[valid_mask]
        target_idxs = neighbor_indices[valid_mask]

        # Update the adjacency matrix and return edge list
        A_t.index_put_((source_idxs, target_idxs), torch.tensor(1.0, device=A_t.device))
        A_t.index_put_((target_idxs, source_idxs), torch.tensor(1.0, device=A_t.device))
        if len(source_idxs) > 0:
            dynamic_edges = torch.stack([source_idxs, target_idxs], dim=0)
        else:
            dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=A_t.device)

    else:
        dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=A_t.device)

    # Normalize new adjacency matrix TODO NOTE: ROW WISE
    row_sums = A_t.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A_t / row_sums

    return A_norm, dynamic_edges


def add_w_edges_radius(base_A, node_types, pos_t, radius):
    """
    Computes A_t dynamically using radius search.
    Excludes existing mesh edges (base_A) and self-loops.

    :param base_A
    :param node_types
    :param pos_t
    :param radius

    :return: (A_norm, dynamic_edges)
        A_norm: Normalized adjacency matrix (including mesh edges + world edges + self loops)
        dynamic_edges: Edge list of ONLY the newly added world edges (2, E_world)
    """
    # Ensure devices match (likely CPU in Dataset)
    if base_A.device != pos_t.device:
        base_A = base_A.to(pos_t.device)

    # Compute pairwise distances, mask radius and esclude self loops
    dists = torch.cdist(pos_t, pos_t)
    radius_mask = dists < radius
    radius_mask.fill_diagonal_(False)

    # Exclude existing mesh edges
    mesh_edge_mask = base_A > 0
    valid_world_mask = radius_mask & (~mesh_edge_mask)

    # Build combined adjacency and normalize
    binary_mesh = mesh_edge_mask.float()
    binary_world = valid_world_mask.float()
    A_combined = binary_mesh + binary_world
    row_sums = A_combined.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1.0
    A_norm = A_combined / row_sums

    # Extract edge list for world edges (for return)
    dynamic_edges = torch.nonzero(binary_world, as_tuple=False).t()
    if dynamic_edges.numel() == 0:
         dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=pos_t.device)

    return A_norm, dynamic_edges


class DefPlateDataset(Dataset):
    total_comp_time = 0.0
    total_comp_calls = 0

    def add_w_edges(self, base_A, node_types, pos_t):
        # Add world edges
        if self.add_world_edges == "radius":
            A_dynamic, dynamic_edges = add_w_edges_radius(base_A=base_A, node_types=node_types, pos_t=pos_t,
                                                          radius=self.radius)
        elif self.add_world_edges == "neighbours":
            A_dynamic, dynamic_edges = add_w_edges_neigh(base_A=base_A, node_types=node_types, pos_t=pos_t,
                                                         k=self.k_neighb)
        elif self.add_world_edges == "None":
            A_dynamic = base_A
            dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=base_A.device)
        else:
            raise ValueError(f"f[defplate_dataset] Wrong add world edges param specified = {self.add_world_edges}"
                             f"choose either 'None' | 'neighbours' | 'radius'")

        return A_dynamic, dynamic_edges

    def __init__(self, list_of_trajs, add_world_edges, k_neighb, radius, world_pos_idxs, velocity_idxs):
        """
        Construct a dataset from a list of trajectories objects.

        :param list_of_trajs: List
            a list where each item has (A, X_seq_norm, mean, std, cells, node_type)
        :param add_world_edges: bool
            whether to add world edges dynamically based on radius search
        :param k_neighb

        :param radius

        """
        self.samples = []
        # Store the list of trajectories
        self.trajs = list_of_trajs
        self.add_world_edges = add_world_edges
        self.k_neighb = k_neighb
        self.radius = radius
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
        Fetches the sample and computes the dynamic adjacency matrix on-the-fly.
        """
        # Retrieve trajectory and its features
        s = self.samples[idx]
        traj_id = s["traj_id"]
        t = s["time_idx"]
        traj = self.trajs[traj_id]
        X_t = traj["X_seq_norm"][t]
        X_tp1 = traj["X_seq_norm"][t + 1]
        base_A = traj["A"]
        node_types = traj["node_type"]

        # time tracking
        time_start = time.time()

        time_start = time.time()
        # Compute Dynamic Adjacency
        pos_t = X_t[:, self.world_pos_idxs]
        # Time tracking ends
        compute_duration = time.time() - time_start

        A_dynamic, dynamic_edges = self.add_w_edges(base_A, node_types, pos_t)

        # Pass all inputs to model: mesh_pos, world_pos, node_type, vel, stress
        # New: Add next-step velocity for kinematic nodes (SPHERE_NODE) as input
        # Extract velocity at t+1 (already normalized)
        v_tp1 = X_tp1[:, self.velocity_idxs]
        # Identify sphere nodes
        sphere_mask = (node_types == SPHERE_NODE)
        # Create feature initialized to 0
        kinematic_vel = torch.zeros_like(v_tp1)
        # Fill only sphere nodes
        kinematic_vel[sphere_mask] = v_tp1[sphere_mask]
        
        # Concatenate to X_t
        X_t_input = torch.cat([X_t, kinematic_vel], dim=-1)

        return (A_dynamic, X_t_input, X_tp1, traj["mean"], traj["std"], traj["cells"], node_types, dynamic_edges,
            traj_id, t, compute_duration)


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
    compute_times_list = []

    for A, X_t, X_tp1, mean, std, cells, node_type, dyn_edges, traj_id, time_idx, comp_time in batch:
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
        compute_times_list.append(comp_time)

    return (adjacency_mat_list, X_t_list, X_tp1_list, mean_list, std_list, cells_list,
        node_types_list, dynamic_edges_list, traj_id_list, time_idx_list, compute_times_list)