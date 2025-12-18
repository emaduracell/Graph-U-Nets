import torch

NORMAL_NODE_OH = [0, 0]  # value 0 (NORMAL)
NORMAL_NODE = 0
SPHERE_NODE_OH = [1, 0]  # value 1 (SPHERE)
SPHERE_NODE = 1
BOUNDARY_NODE_OH = [0, 1]  # value 3 (BOUNDARY)
BOUNDARY_NODE = 3
VELOCITY_MEAN = 0.0

def add_w_edges_neigh(base_A, node_types, pos_t, k):
    """
    For each Sphere node, it looks at its k-nearest neighbors.
    If a neighbor is a Plate node (Normal or Boundary), it adds an edge.
    If a neighbor is another Sphere node, it ignores it (assumed already handled or irrelevant).

    Args:
        base_A
        node_types
        pos_t
        radius

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

    return A_t, dynamic_edges


def add_w_edges_radius(base_A, node_types, pos_t, radius):
    """
    Computes A_t dynamically using radius search.
    Excludes existing mesh edges (base_A) and self-loops.

    Args:
        base_A
        node_types
        pos_t
        radius

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

    # Exclude sphere-sphere interactions
    is_sphere = (node_types.to(pos_t.device) == SPHERE_NODE).view(-1)
    sphere_sphere_mask = is_sphere.unsqueeze(1) & is_sphere.unsqueeze(0)

    valid_world_mask = radius_mask & (~mesh_edge_mask) & (~sphere_sphere_mask)

    # Build combined adjacency and normalize
    binary_mesh = mesh_edge_mask.float()
    binary_world = valid_world_mask.float()
    A_combined = binary_mesh + binary_world

    # Extract edge list for world edges (for return)
    dynamic_edges = torch.nonzero(binary_world, as_tuple=False).t()
    if dynamic_edges.numel() == 0:
         dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=pos_t.device)

    return A_combined, dynamic_edges


def add_w_edges(edge_config, base_A, node_types, pos_t):
    """
    Add world edges based on configuration.
    
    Args:
        edge_config: SimpleNamespace or object with attributes:
            - add_world_edges: str ('radius', 'k_neighb', or 'None')
            - radius: float (if add_world_edges == 'radius')
            - k_neighb: int (if add_world_edges == 'k_neighb')
        base_A: torch.Tensor
            Base adjacency matrix
        node_types: torch.Tensor
            Node type tensor
        pos_t: torch.Tensor
            Position tensor at time t
    
    :return: (A_dynamic, dynamic_edges)
    """
    # Add world edges
    if edge_config.add_world_edges == "radius":
        A_dynamic, dynamic_edges = add_w_edges_radius(base_A=base_A, node_types=node_types, pos_t=pos_t,
                                                      radius=edge_config.radius)
    elif edge_config.add_world_edges == "k_neighb":
        A_dynamic, dynamic_edges = add_w_edges_neigh(base_A=base_A, node_types=node_types, pos_t=pos_t,
                                                     k=edge_config.k_neighb)
    elif edge_config.add_world_edges == "None":
        A_dynamic = base_A
        dynamic_edges = torch.empty((2, 0), dtype=torch.long, device=base_A.device)
    else:
        raise ValueError(f"[add_world_edges] Wrong add_world_edges value add_world_edges = {edge_config.add_world_edges}, "
                         f"choose either 'None' | 'k_neighb' | 'radius'")

    return A_dynamic, dynamic_edges
