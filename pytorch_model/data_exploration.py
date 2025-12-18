import torch
import numpy as np
import os
import plotly.graph_objects as go
from data.add_world_edges import add_w_edges_radius

# Constants
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1


def make_dynamic_edges_trace(coords, edge_index):
    """
    Creates Red Lines for the dynamic interactions (world edges).

    Args:
        coords: Numpy array or tensor containing node coordinates (N, 3).
        edge_index: Tensor or array containing connectivity indices for edges (2, E).
    :return Scatter3d
    """
    if edge_index is None:
        return go.Scatter3d()

    if isinstance(edge_index, torch.Tensor):
        if edge_index.shape[1] == 0:
            return go.Scatter3d()
        src = edge_index[0].cpu().numpy()
        dst = edge_index[1].cpu().numpy()
    else:
        if edge_index.shape[1] == 0:
            return go.Scatter3d()
        src = edge_index[0]
        dst = edge_index[1]

    x_lines, y_lines, z_lines = [], [], []

    # Vectorized construction for Plotly lines
    nan_vec = np.full(src.shape, None)

    for dim, lines_list in enumerate([x_lines, y_lines, z_lines]):
        c_src = coords[src, dim]
        c_dst = coords[dst, dim]
        # Stack
        stacked = np.stack([c_src, c_dst, nan_vec], axis=1).flatten()
        lines_list.extend(stacked)

    return go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines', line=dict(color='red', width=4),
                        name='World Edges (Ground Truth)')


def make_wireframe(x, y, z, i, j, k, color='black', width=1.5):
    """
    Creates a Scatter3d trace that draws the edges of the triangles.

    Args:
        x: X-coordinates of the mesh nodes.
        y: Y-coordinates of the mesh nodes.
        z: Z-coordinates of the mesh nodes.
        i: Indices of the first vertices of the triangular elements.
        j: Indices of the second vertices of the triangular elements.
        k: Indices of the third vertices of the triangular elements.
        color: Color string for the wireframe lines.
        width: Float specifying the width of the wireframe lines.
    :return: Scatter3d trace
    """
    tri_points = np.vstack([
        i, j, k, i,
        np.full_like(i, -1)
    ]).T.flatten()

    xe = x[tri_points]
    ye = y[tri_points]
    ze = z[tri_points]

    xe[4::5] = None
    ye[4::5] = None
    ze[4::5] = None

    return go.Scatter3d(x=xe, y=ye, z=ze, mode='lines', line=dict(color=color, width=width), name='Wireframe',
                        showlegend=False, hoverinfo='skip')


def print_dataset_statistics(metadata_path, world_pos_idxs, vel_idxs, stress_idxs, mesh_pos_idxs=None):
    """
    Loads and prints the statistics calculated by data_loader and saved by main_data.

    Args:
        metadata_path: File path to the saved metadata dictionary (.pt file).
        world_pos_idxs: Slices or indices corresponding to world position features.
        vel_idxs: Slices or indices corresponding to velocity features.
        stress_idxs: Slices or indices corresponding to stress features.
        mesh_pos_idxs: Optional slices or indices corresponding to mesh position features.
    """
    print("\n" + "=" * 60)
    print(" DATASET STATISTICS (Calculated by Data Loader)")
    print("=" * 60)

    if not os.path.exists(metadata_path):
        print(f"Warning: Metadata file not found at {metadata_path}")
        return

    meta = torch.load(metadata_path)

    # Extract mean and std
    mean_vec = meta['mean'].squeeze()
    std_vec = meta['std'].squeeze()

    print(f"Total Trajectories: {meta['num_trajectories']}")
    print(f"Time Steps per Traj: {meta['time_steps']}")
    print(f"Nodes per Traj:      {meta['num_nodes']}")
    print("-" * 60)
    print(f"{'FEATURE':<20} | {'MEAN':<25} | {'STD DEV':<25}")
    print("-" * 60)

    # Helper to print vectors
    def fmt(vec):
        return f"[{', '.join([f'{x:.4f}' for x in vec])}]"

    # World Position
    w_mean = mean_vec[world_pos_idxs]
    w_std = std_vec[world_pos_idxs]
    print(f"{'World Pos (X,Y,Z)':<20} | {fmt(w_mean):<25} | {fmt(w_std):<25}")

    # Mesh Position (if applicable)
    if mesh_pos_idxs is not None:
        m_mean = mean_vec[mesh_pos_idxs]
        m_std = std_vec[mesh_pos_idxs]
        print(f"{'Mesh Pos (X,Y,Z)':<20} | {fmt(m_mean):<25} | {fmt(m_std):<25}")

    # Velocity
    v_mean = mean_vec[vel_idxs]
    v_std = std_vec[vel_idxs]
    print(f"{'Velocity (Vx,Vy,Vz)':<20} | {fmt(v_mean):<25} | {fmt(v_std):<25}")

    # Stress
    s_mean = mean_vec[stress_idxs]
    s_std = std_vec[stress_idxs]
    print(f"{'Stress (Von Mises)':<20} | {fmt(s_mean):<25} | {fmt(s_std):<25}")

    print("-" * 60)
    print("Note: If normalization_method='centroid', positions and velocities are centered per-frame.")
    print("The statistics above represent the global normalization scaling factors.")
    print("=" * 60 + "\n")


def visualize_ground_truth(pos, cells, stress, node_type, title, color_mode="stress", dynamic_edges=None):
    """
    Visualizes a single state of the ground truth data.

    Args:
        pos: Array of node positions with shape (N, 3).
        cells: Array containing cell connectivity (tetrahedra or triangles).
        stress: Array of stress values for each node.
        node_type: Array of node types (e.g., normal, boundary).
        title: String title for the Plotly figure.
        color_mode: String indicating the coloring mode ('stress' or 'node_type').
        dynamic_edges: Optional edge index for drawing world/dynamic edges.
    """
    # Triangulation logic
    tri_i, tri_j, tri_k = [], [], []
    for (i0, i1, i2, i3) in cells:
        tri_i.extend([i0, i0, i0, i1])
        tri_j.extend([i1, i2, i3, i3])
        tri_k.extend([i2, i3, i1, i2])

    # Color setup
    if color_mode == "stress":
        intensity = stress if stress is not None else np.zeros(pos.shape[0])
        colorscale = "Viridis"
        cbar_title = "Stress"
    elif color_mode == "node_type":
        intensity = node_type if node_type is not None else np.zeros(pos.shape[0])
        colorscale = "Turbo"
        cbar_title = "Type"
    else:
        raise ValueError("Unknown color_mode")

    fig = go.Figure()

    # Surface Mesh
    fig.add_trace(go.Mesh3d(
        x=pos[:, 0], y=pos[:, 1], z=pos[:, 2],
        i=tri_i, j=tri_j, k=tri_k,
        intensity=intensity,
        colorscale=colorscale,
        showscale=True,
        colorbar=dict(title=cbar_title),
        flatshading=True,
        opacity=0.9,
        name="Surface"
    ))

    # Wireframe
    fig.add_trace(make_wireframe(pos[:, 0], pos[:, 1], pos[:, 2], np.array(tri_i), np.array(tri_j), np.array(tri_k)))

    # Give the right side more room for the colorbar
    fig.update_layout(margin=dict(l=0, r=160, t=60, b=0))

    # Move legend away from the colorbar (e.g., top-left)
    fig.update_layout(legend=dict(x=0.01, y=0.99, xanchor="left", yanchor="top", bgcolor="rgba(255,255,255,0.6)"))

    # Push the mesh colorbar further right
    fig.update_traces(selector=dict(type="mesh3d"), colorbar=dict(x=1.08))

    # Dynamic Edges (World Edges)
    if dynamic_edges is not None:
        fig.add_trace(make_dynamic_edges_trace(pos, dynamic_edges))

    fig.update_scenes(aspectmode="data")
    fig.update_layout(height=700, width=900, title_text=title, scene=dict(
        xaxis_title="X", yaxis_title="Y", zaxis_title="Z"
    )
                      )
    fig.show()


def apply_filter_mask(pos, stress, node_type, cells, render_mode):
    """
    Filters nodes based on render_mode (e.g., hiding borders).

    Args:
        pos: Array of node positions.
        stress: Array of node stress values.
        node_type: Array of node types.
        cells: Array of cell connectivity indices.
        render_mode: String specifying the rendering filter (e.g., 'all', 'no_border').
    """
    mode = render_mode.lower()
    if mode == "all":
        return pos, stress, node_type, cells

    mask = np.ones(node_type.shape[0], dtype=bool)
    if "no_border" in mode:
        mask &= (node_type != BOUNDARY_NODE)
    if "no_sphere" in mode:
        mask &= (node_type != SPHERE_NODE)

    if not mask.any():
        raise ValueError("Render mask removed all nodes.")

    # Reindex
    idx_map = -np.ones(mask.shape[0], dtype=int)
    keep_idx = np.nonzero(mask)[0]
    idx_map[keep_idx] = np.arange(keep_idx.shape[0])

    # Filter cells
    keep_cells = np.all(mask[cells], axis=1)
    cells_kept = cells[keep_cells]
    cells_reindexed = idx_map[cells_kept]

    return pos[mask], stress[mask], node_type[mask], cells_reindexed


def main(render_mode, traj_idx, t_step, preprocessed_path, metadata_path, add_world_edges, include_mesh_pos):
    """
    Args:
        render_mode: String filter for visualization (e.g., 'all').
        traj_idx: Integer index of the trajectory to load and visualize.
        t_step: Integer index of the time step to extract.
        preprocessed_path: Path to the file containing preprocessed trajectory data.
        metadata_path: Path to the file containing dataset statistics.
        add_world_edges: Parameter relating to world edge inclusion (Note: unused in function body).
        include_mesh_pos: Boolean flag indicating if mesh positions are included in the dataset features.
    """

    if include_mesh_pos:
        mesh_pos_idxs = slice(0, 3)
        world_pos_idxs = slice(3, 6)
        node_type_idxs = slice(6, 8)
        vel_idxs = slice(8, 11)
        stress_idxs = slice(11, 12)
    else:
        mesh_pos_idxs = None
        world_pos_idxs = slice(0, 3)
        node_type_idxs = slice(3, 5)
        vel_idxs = slice(5, 8)
        stress_idxs = slice(8, 9)

    # Print Global Statistics
    print_dataset_statistics(metadata_path, world_pos_idxs, vel_idxs, stress_idxs, mesh_pos_idxs)

    # Load Trajectory Data
    print(f"Loading trajectory {traj_idx} from {preprocessed_path}...")
    if not os.path.exists(preprocessed_path):
        raise ValueError(f"Data not found at {preprocessed_path}\n"
                         f"Please check OUTPUT_DIR_BASE and NORM/MESH settings in main block.")

    list_of_trajs = torch.load(preprocessed_path)
    if traj_idx >= len(list_of_trajs):
        raise ValueError(f"Index {traj_idx} out of bounds (Max {len(list_of_trajs) - 1})")

    traj = list_of_trajs[traj_idx]

    # Unpack tensors
    A = traj["A"]
    if A.ndim == 3:
        A = A[t_step]
    X_seq_norm = traj["X_seq_norm"]
    mean = traj["mean"]
    std = traj["std"]
    cells = traj["cells"]
    node_type = traj["node_type"]

    # Validate Time Step
    T = X_seq_norm.shape[0]
    if t_step >= T:
        print(f"Time step {t_step} exceeds trajectory length {T}. Showing last frame.")
        t_step = T - 1

    # Extract and Denormalize Data for specific step
    X_t_norm = X_seq_norm[t_step]

    # Denormalize: Phys = Norm * Std + Mean
    mean_vec = mean.squeeze()
    std_vec = std.squeeze()

    X_t_phys = X_t_norm * std_vec + mean_vec

    # Extract features using indices
    pos_phys = X_t_phys[:, world_pos_idxs].numpy()
    stress_phys = X_t_phys[:, stress_idxs].numpy().squeeze()

    # Node type is not normalized
    node_type_np = node_type.numpy()
    cells_np = cells.numpy()

    # Filter for Visualization
    pos_viz, stress_viz, type_viz, cells_viz = apply_filter_mask(pos_phys, stress_phys, node_type_np, cells_np,
                                                                 render_mode)

    # Compute Dynamic Edges
    dynamic_edges = None
    if "world_edge_index" in traj:
        print("Using pre-computed world edges from data loader...")
        # Check if world_edge_index is a list (per timestep) or single tensor
        if isinstance(traj["world_edge_index"], list):
            dynamic_edges = traj["world_edge_index"][t_step]
        else:
            dynamic_edges = traj["world_edge_index"]
    else:
        print("No pre-computed world edges found.")

    # Visualize
    print(f"Visualizing Trajectory {traj_idx} at Time Step {t_step}...")
    visualize_ground_truth(pos=pos_viz, cells=cells_viz, stress=stress_viz, node_type=type_viz,
                           title=f"Data Exploration: Traj {traj_idx}, t={t_step}", color_mode="stress",
                           dynamic_edges=dynamic_edges)


if __name__ == "__main__":
    # Choose dataset by setting NORM_METHOD, INCLUDE_MESH_POS
    NORM_METHOD = "standard"
    INCLUDE_MESH_POS = True

    # Hardcoded path to data + metadata
    DATA_DIR = "processed_data/data_radius_True"
    PREPROCESSED_FILE = os.path.join(DATA_DIR, "preprocessed_train.pt")
    METADATA_FILE = os.path.join(DATA_DIR, "preprocessed_metadata.pt")

    # Visualization Settings: change this to your preference
    TRAJ_IDX = 9
    TIME_STEP = 288
    RENDER_MODE = "all"

    main(RENDER_MODE, TRAJ_IDX, TIME_STEP, PREPROCESSED_FILE, METADATA_FILE, NORM_METHOD, INCLUDE_MESH_POS)