import torch
import numpy as np
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data.add_world_edges import add_w_edges_radius
from model_gunet.gunet_deforming_plate import GraphUNet_DefPlate
from data_builder import build_adjacency_matrix
from helpers.helpers import get_feature_indices, load_config

OUTPUT_DIR = "simulation_rollout"
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1
# Data paths
TFRECORD_PATH = "raw_data/train.tfrecord"
META_PATH = "raw_data/meta.json"


def make_dynamic_edges_trace(coords, edge_index):
    """
    Creates Red Lines for the dynamic interactions (world edges).

    Args:
        coords: [N, 3] numpy array of node coordinates.
        edge_index: [2, E] tensor or numpy array defining source and destination nodes.
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

    nan_vec = np.full(src.shape, None)

    for dim, lines_list in enumerate([x_lines, y_lines, z_lines]):
        c_src = coords[src, dim]
        c_dst = coords[dst, dim]
        # Stack: [E, 3] where cols are src, dst, None
        stacked = np.stack([c_src, c_dst, nan_vec], axis=1).flatten()
        lines_list.extend(stacked)

    return go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines',
                        line=dict(color='red', width=4),  # Thick red lines
                        name='World Edges')


def make_wireframe(x, y, z, i, j, k, color='black', width=1.5):
    """
    Creates a Scatter3d trace that draws the edges of the triangles.

    Args:
        x, y, z: Arrays containing the x, y, and z coordinates of the vertices.
        i, j, k: Arrays containing the indices of the vertices for each triangle.
        color: String specifying the color of the wireframe lines.
        width: Float specifying the width of the wireframe lines.
    """

    tri_points = np.vstack([
        i, j, k, i,
        np.full_like(i, -1)  # Placeholder for None
    ]).T.flatten()

    # Map indices to coordinates
    xe = x[tri_points]
    ye = y[tri_points]
    ze = z[tri_points]

    # Insert None where we had the -1 index to break the lines
    xe[4::5] = None
    ye[4::5] = None
    ze[4::5] = None

    return go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode='lines',
        line=dict(color=color, width=width),
        name='wireframe',
        showlegend=False,
        hoverinfo='skip'
    )


# Wrapper for model args
class ArgsWrapper:
    """
    Simple wrapper class to emulate an arguments object for storing model hyperparameters.
    """
    pass


def visualize_mesh_pair(pos_true, pos_pred, cells, stress_true, stress_pred, node_type_true, node_type_pred, title_true,
                        title_pred, color_mode, dynamic_edges):
    """
    Visualizzazione mesh + heatmap stress o node_type.

    Args:
        pos_true: [N, 3] Numpy array of ground truth node positions.
        pos_pred: [N, 3] Numpy array of predicted node positions.
        cells: List or array of cell indices defining the mesh connectivity.
        stress_true: [N] Numpy array of ground truth stress values.
        stress_pred: [N] Numpy array of predicted stress values.
        node_type_true: [N] Numpy array of ground truth node types.
        node_type_pred: [N] Numpy array of predicted node types.
        title_true: String title for the ground truth subplot.
        title_pred: String title for the prediction subplot.
        color_mode: String, either "stress" or "node_type" to determine coloring strategy.
        dynamic_edges: [2, E] Array of edge indices for dynamic world edges to visualize.
    """

    if pos_true.ndim == 3:
        raise ValueError("pos_true should not have a batch dimension")
    if pos_pred.ndim == 3:
        raise ValueError("pos_pred should not have a batch dimension")

    if stress_true is not None and stress_true.ndim == 2:
        raise ValueError("stress_true should not have a batch dimension")
    if stress_pred is not None and stress_pred.ndim == 2:
        raise ValueError("stress_pred should not have a batch dimension")

    if node_type_true is not None and node_type_true.ndim == 2:
        raise ValueError("node_type_true should not have a batch dimension")
    if node_type_pred is not None and node_type_pred.ndim == 2:
        raise ValueError("node_type_pred should not have a batch dimension")

    tri_i, tri_j, tri_k = [], [], []
    for (i0, i1, i2, i3) in cells:
        # Face 1: Bottom
        tri_i.extend([i0])
        tri_j.extend([i1])
        tri_k.extend([i2])

        # Face 2: Side A
        tri_i.extend([i0])
        tri_j.extend([i2])
        tri_k.extend([i3])

        # Face 3: Side B
        tri_i.extend([i0])
        tri_j.extend([i3])
        tri_k.extend([i1])

        # Face 4: Back/Front (Base 2)
        tri_i.extend([i1])
        tri_j.extend([i3])
        tri_k.extend([i2])

    # Stress mode
    if color_mode == "stress":
        if stress_true is None:
            intensity_true = np.zeros(pos_true.shape[0])
        else:
            intensity_true = stress_true.astype(float)

        if stress_pred is None:
            intensity_pred = np.zeros(pos_pred.shape[0])
        else:
            intensity_pred = stress_pred.astype(float)

        colorscale = "Viridis"

    # Node type mode
    elif color_mode == "node_type":
        if node_type_true is None:
            intensity_true = np.zeros(pos_true.shape[0])
        else:
            intensity_true = node_type_true.astype(float)

        if node_type_pred is None:
            intensity_pred = np.zeros(pos_pred.shape[0])
        else:
            intensity_pred = node_type_pred.astype(float)

        # discrete but continuous scale
        colorscale = "Turbo"

    else:
        raise ValueError("color_mode must be 'stress' or 'node_type'")

    # Figure setup
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(title_true, title_pred)
    )

    # Surface mesh
    fig.add_trace(
        go.Mesh3d(x=pos_true[:, 0], y=pos_true[:, 1], z=pos_true[:, 2],
                  i=tri_i, j=tri_j, k=tri_k,
                  intensity=intensity_true,
                  colorscale=colorscale,
                  showscale=True,
                  flatshading=True,
                  opacity=0.85,
                  name="true_mesh"
                  ),
        row=1, col=1
    )
    # The Wireframe
    fig.add_trace(make_wireframe(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                                 np.array(tri_i), np.array(tri_j), np.array(tri_k)), row=1, col=1)

    # The Surface
    fig.add_trace(
        go.Mesh3d(x=pos_pred[:, 0], y=pos_pred[:, 1], z=pos_pred[:, 2], i=tri_i, j=tri_j, k=tri_k,
                  intensity=intensity_pred, colorscale=colorscale, showscale=True, flatshading=True,
                  opacity=0.85, name="pred_mesh"), row=1, col=2
    )
    # The Wireframe
    fig.add_trace(make_wireframe(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
                                 np.array(tri_i), np.array(tri_j), np.array(tri_k)), row=1, col=2)

    if dynamic_edges is not None:
        fig.add_trace(make_dynamic_edges_trace(pos_pred, dynamic_edges), row=1, col=2)

    fig.update_scenes(aspectmode="data")
    fig.update_layout(height=600, width=1200, title_text="Mesh Comparison")
    fig.show()


def apply_render_mode(pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells):
    """
    Optionally drop border and/or sphere nodes before rendering based on RENDER_MODE.

    Args:
        pos_true: Ground truth positions.
        pos_pred: Predicted positions.
        stress_true: Ground truth stress.
        stress_pred: Predicted stress.
        node_type_true: Ground truth node types.
        node_type_pred: Predicted node types.
        cells: Mesh connectivity indices.

    Returns:
        Filtered copies of inputs with remapped cell indices.
    """
    mode = render_mode.lower()
    if mode == "all":
        return pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells

    mask = np.ones(node_type_true.shape[0], dtype=bool)
    if "no_border" in mode:
        mask &= (node_type_true != BOUNDARY_NODE)
    if "no_sphere" in mode:
        mask &= (node_type_true != SPHERE_NODE)

    if not mask.any():
        raise ValueError("Render mask removed all nodes. Adjust RENDER_MODE.")

    # Reindex nodes
    idx_map = -np.ones(mask.shape[0], dtype=int)
    keep_idx = np.nonzero(mask)[0]
    idx_map[keep_idx] = np.arange(keep_idx.shape[0])

    # Filter cells to ones fully kept, then remap their indices
    keep_cells = np.all(mask[cells], axis=1)
    cells_kept = cells[keep_cells]
    cells_reindexed = idx_map[cells_kept]

    # Apply mask to per-node arrays
    pos_true_f = pos_true[mask]
    pos_pred_f = pos_pred[mask]
    stress_true_f = stress_true[mask] if stress_true is not None else None
    stress_pred_f = stress_pred[mask] if stress_pred is not None else None
    node_type_true_f = node_type_true[mask] if node_type_true is not None else None
    node_type_pred_f = node_type_pred[mask] if node_type_pred is not None else None

    return pos_true_f, pos_pred_f, stress_true_f, stress_pred_f, node_type_true_f, node_type_pred_f, cells_reindexed


def rollout(model, A, X_seq_norm, mean_vec, std_vec, t0, steps, node_type, vel_idxs, stress_idxs, node_type_idxs,
            world_pos_idxs, add_world_edges, radius):
    """
    Autoregressive rollout that:
      - predicts plate velocities + stresses,
      - keeps borders fixed (node_type == 3),
      - drives rigid body (node_type == 1) with scripted (ground-truth) motion.

    Args:
        model: GraphUNet_DefPlate model instance.
        A: Tensor [N,N]
            Adjacency matrix.
        X_seq_norm: Tensor [T,N,F]
            Normalized feature sequence for this trajectory.
        mean, std: Tensors [1,1,F]
            Normalization stats.
        t0: int
            Starting time index.
        steps: int
            Number of rollout steps.
        node_type: Tensor [N]
            Integer node types.
        vel_idxs: list or slice
            Indices corresponding to velocity features in the input vector.
        stress_idxs: list or slice
            Indices corresponding to stress features in the input vector.
        node_type_idxs: list or slice
            Indices corresponding to node type features in the input vector.
        world_pos_idxs: list or slice
            Indices corresponding to world position features in the input vector.
        add_world_edges: bool
            Flag to enable dynamic world edge computation during rollout.
        radius: float
            Radius threshold for creating dynamic world edges.

    Returns:
        coords_pred_list: list of [N,3] np.arrays of predicted coordinates.
        stress_pred_list: list of [N] np.arrays of predicted stress values.
        node_type_pred_list: list of [N] np.arrays of node types.
        rollout_error_list: list of MSE values per step.
        dynamic_edges_list: list of edge indices used at each step.
    """

    device = A.device
    mean_vec = mean_vec.to(device)
    std_vec = std_vec.to(device)
    node_type = node_type.to(device)
    # Node type features must follow data_loader encoding: [sphere, boundary]
    node_type_onehot = torch.zeros((node_type.shape[0], 2), device=device, dtype=mean_vec.dtype)
    node_type_onehot[:, 0] = (node_type == SPHERE_NODE).float()
    node_type_onehot[:, 1] = (node_type == BOUNDARY_NODE).float()

    # Masks
    deform_mask = (node_type == NORMAL_NODE)  # deformable plate
    rigid_mask = (node_type == SPHERE_NODE)  # rigid body
    border_mask = (node_type == BOUNDARY_NODE)  # fixed borders

    # initial state at t_0
    current_norm = X_seq_norm[t0].to(device)
    if current_norm.dim() == 3:
        current_norm = current_norm[0]

    current_phys = current_norm * std_vec + mean_vec
    # ground truth at t0
    p_hat = current_phys[:, world_pos_idxs].clone()

    # Borders reference positions (fixed in time)
    pos_border_ref = p_hat[border_mask].clone()

    coords_pred_list = []
    stress_pred_list = []
    node_type_pred_list = []
    rollout_error_list = []
    base_A = A.clone()
    dynamic_edges_list = []  # Store edges for viz
    for k in range(steps):
        # Generate world edges for the current predicted state
        if add_world_edges:
            A_dynamic, dyn_edges = add_w_edges_radius(base_A, node_type, p_hat, radius=radius)
        else:
            A_dynamic = base_A
            dyn_edges = None

        # print(f"dyn_edges={dyn_edges.shape}")
        dynamic_edges_list.append(dyn_edges)

        with torch.no_grad():
            pred = model.rollout_step(A_dynamic, current_norm)

        vel_norm = pred[:, :3]  # [N,3]
        stress_norm = pred[:, 3].unsqueeze(-1)

        # Denormalize predicted velocity & stress (v_hat_k, sigma_hat_k)
        vel_pred = vel_norm * std_vec[vel_idxs] + mean_vec[vel_idxs]
        stress_pred = stress_norm * std_vec[stress_idxs] + mean_vec[stress_idxs]

        # Start p_hat_{k+1} as a copy of p_hat_k
        p_hat_next = p_hat.clone()
        stress_next = stress_pred.clone()

        # deformable plate nodes (node_type == 0)  p_hat_{k+1} = p_hat_k + v_hat_k  (for deformable nodes)
        p_hat_next[deform_mask] = p_hat[deform_mask] + vel_pred[deform_mask]

        # Rigid body nodes: follow scripted (ground-truth) motion at time t0 + 1 + k
        gt_norm_step = X_seq_norm[t0 + 1 + k].to(device)
        if gt_norm_step.dim() == 3:
            raise ValueError("gt_norm_step should not have a batch dimension")

        gt_phys_step = gt_norm_step * std_vec + mean_vec
        p_rigid_gt = gt_phys_step[:, :3]
        v_rigid_gt = gt_phys_step[:, vel_idxs]
        s_rigid_gt = gt_phys_step[:, stress_idxs]

        # drive rigid nodes with GT
        p_hat_next[rigid_mask] = p_rigid_gt[rigid_mask]
        vel_pred[rigid_mask] = v_rigid_gt[rigid_mask]
        stress_next[rigid_mask] = s_rigid_gt[rigid_mask]

        # Fixed borders
        p_hat_next[border_mask] = pos_border_ref
        vel_pred[border_mask] = 0.0  # explicitly zero velocity

        # Store p_k hat
        coords_pred_list.append(p_hat_next.detach().cpu().numpy())
        stress_pred_list.append(stress_next.detach().cpu().numpy())
        node_type_pred_list.append(node_type.detach().cpu().numpy())
        p_gt_next = gt_phys_step[:, :3]
        # Calculate MSE: mean((Pred - True)^2)
        mse_step = torch.mean((p_hat_next - p_gt_next) ** 2)
        rollout_error_list.append(mse_step.item())

        # Build physical features X_{k+1} from (p_hat_{k+1}, v_hat_k+rigid/border overrides)
        X_next_phys = torch.zeros_like(current_phys)
        X_next_phys[:, world_pos_idxs] = p_hat_next
        if world_pos_idxs.start > 0:
            X_next_phys[:, :world_pos_idxs.start] = current_phys[:, :world_pos_idxs.start]
        X_next_phys[:, node_type_idxs] = node_type_onehot  # node type one-hot
        X_next_phys[:, vel_idxs] = vel_pred  # velocity field
        X_next_phys[:, stress_idxs] = stress_next  # stress

        # Re-normalize for next model input (graph at time k+1)
        current_phys = X_next_phys
        current_norm = (X_next_phys - mean_vec) / std_vec

        # Advance p_hat_k -> p_hat_{k+1}
        p_hat = p_hat_next

    return coords_pred_list, stress_pred_list, node_type_pred_list, rollout_error_list, dynamic_edges_list


def main(mesh_pos_idxs, world_pos_idxs, node_type_idxs, vel_idxs, stress_idxs, dim_in, render_mode, rollout_steps,
         traj_idx, t_step, rollout_set, preprocessed_path, add_world_edges, checkpoint_path):
    """
    Main execution function to load data, load the model, and perform visualization or rollout.

    Args:
        mesh_pos_idxs: Indices for mesh position features.
        world_pos_idxs: Indices for world position features.
        node_type_idxs: Indices for node type features.
        vel_idxs: Indices for velocity features.
        stress_idxs: Indices for stress features.
        dim_in: Input dimension for the neural network.
        render_mode: String specifying rendering filter ('all', 'no_border', etc.).
        rollout_steps: Number of steps to simulate in the rollout.
        traj_idx: Index of the trajectory to load from the dataset.
        t_step: Initial time step index for the simulation.
        rollout_set: Boolean, if True runs a multi-step rollout, else single step.
        preprocessed_path: Path to the preprocessed dataset file.
        add_world_edges: Boolean, determines if dynamic edges are calculated.
        checkpoint_path: Path to the saved model checkpoint.
    """

    # Load data
    print("Loading trajectory...")
    # Prefer preprocessed data (same as training); fall back to raw TFRecord if missing
    if os.path.exists(preprocessed_path):
        list_of_trajs = torch.load(preprocessed_path)
        if traj_idx >= len(list_of_trajs):
            raise ValueError(f"Requested TRAJ_INDEX={traj_idx} but only {len(list_of_trajs)} trajectories in "
                             f"{preprocessed_path}")
        # Keep only what we need to avoid extra host->device transfers
        list_of_trajs = list_of_trajs[:traj_idx + 1]
        print(f"Loaded {len(list_of_trajs)} preprocessed trajectories from {preprocessed_path}")
    else:
        raise ValueError(f"Preprocessed data not found at {preprocessed_path}")
    traj = list_of_trajs[traj_idx]

    A = traj["A"]
    X_seq_norm = traj["X_seq_norm"]
    mean = traj["mean"]
    std = traj["std"]
    cells = traj["cells"]
    node_type = traj["node_type"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reconstruct clean base_A from cells for rollout.
    num_nodes = X_seq_norm.shape[1]
    if isinstance(cells, torch.Tensor):
        cells_np = cells.cpu().numpy()
    else:
        cells_np = cells
    base_A_clean = build_adjacency_matrix(cells_np, num_nodes).to(device)
    A = base_A_clean

    X_seq_norm = X_seq_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)
    node_type = node_type.to(device)

    # Select time step
    T = X_seq_norm.shape[0]
    t = t_step
    if not (0 <= t < T - 1):
        raise ValueError(f"t must be in [0, {T - 2}]")

    X_t_norm = X_seq_norm[t]
    X_tp_norm = X_seq_norm[t + 1]

    # remove batch dim if needed
    if X_t_norm.dim() == 3:
        raise ValueError("X_t_norm should not have a batch dimension")
    if X_tp_norm.dim() == 3:
        raise ValueError("X_tp_norm should not have a batch dimension")

    # Build model
    # Load model hyperparameters from the same YAML used in training
    config_path = os.path.join(os.path.dirname(__file__), "config_gunet.yaml")
    config = load_config(config_path)
    model_cfg = config["model"]

    myargs = ArgsWrapper()
    myargs.activation_gnn = model_cfg["activation_gnn"]
    myargs.activation_mlps_final = model_cfg["activation_mlps_final"]
    myargs.hid_gnn_layer_dim = model_cfg["hid_gnn_layer_dim"]
    myargs.hid_mlp_dim = model_cfg["hid_mlp_dim"]
    myargs.k_pool_ratios = model_cfg["k_pool_ratios"]
    myargs.dropout_gnn = model_cfg["dropout_gnn"]
    myargs.dropout_mlps_final = model_cfg["dropout_mlps_final"]

    # dim_in = X_seq_norm.shape[2]
    # Model trained to output [vx,vy,vz,stress]
    model = GraphUNet_DefPlate(dim_in, 3, 1, myargs, adj_norm=model_cfg['adj_norm']).to(device)
    state = torch.load(checkpoint_path, map_location=device)

    # Backwards compatibility: old checkpoints used "s_gcn" instead of "start_gcn"
    if "s_gcn.proj.weight" in state:
        state["start_gcn.proj.weight"] = state.pop("s_gcn.proj.weight")
    if "s_gcn.proj.bias" in state:
        state["start_gcn.proj.bias"] = state.pop("s_gcn.proj.bias")
    model.load_state_dict(state)
    model.eval()

    # One step prediction
    # Ground-truth coordinates and von Mises stress at t+1 (for comparison)
    mean_vec = mean[0, 0]
    std_vec = std[0, 0]

    coords_true = X_tp_norm[:, :3] * std_vec[:3] + mean_vec[:3]
    pos_true = coords_true.cpu().numpy()

    stress_true = X_tp_norm[:, stress_idxs] * std_vec[stress_idxs] + mean_vec[stress_idxs]
    stress_true = stress_true.cpu().numpy().squeeze(-1)

    # Use rollout with 1 step to integrate predicted velocities
    coords_pred_list, stress_pred_list, node_type_pred_list, rollout_error_list, dynamic_edges_list = \
        rollout(model=model, A=A, X_seq_norm=X_seq_norm, mean_vec=mean_vec, std_vec=std_vec, t0=t, steps=1,
                node_type=node_type, vel_idxs=vel_idxs, stress_idxs=stress_idxs, node_type_idxs=node_type_idxs,
                world_pos_idxs=world_pos_idxs, add_world_edges=add_world_edges,
                radius=dataconfig.get('radius_world_edge', 0.03))

    pos_pred = coords_pred_list[0]
    stress_pred = stress_pred_list[0].squeeze(-1)

    # Node types for visualization: always use ground-truth integers
    node_type_np = node_type.cpu().numpy()
    node_type_true = node_type_np
    node_type_pred = node_type_np

    if isinstance(cells, torch.Tensor):
        cells = cells.cpu().numpy()

    # Single step
    pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells_filtered = apply_render_mode(
        pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells
    )

    if not rollout_set:
        base_A_tensor = A
        node_type_tensor = node_type
        pos_pred_tensor = torch.tensor(pos_pred, device=device, dtype=torch.float32)

        if add_world_edges:
            _, dynamic_edges_single = add_w_edges_radius(base_A_tensor, node_type_tensor, pos_pred_tensor, radius=0.03)
        else:
            dynamic_edges_single = None

        visualize_mesh_pair(pos_true=pos_true, pos_pred=pos_pred, stress_true=stress_true, stress_pred=stress_pred,
                            node_type_true=node_type_true, node_type_pred=node_type_pred, cells=cells_filtered,
                            color_mode="stress", title_true=f"Ground Truth t={t + 1}",
                            title_pred=f"Prediction t={t + 1}", dynamic_edges=dynamic_edges_single)
        return

    # Multi step rollout
    steps = min(rollout_steps, T - 1 - t)
    print(f"\nPerforming {steps}-step rollout...")

    coords_pred_list, stress_pred_list, node_type_pred_list, rollout_error_list, dynamic_edges_list = \
        rollout(model=model, A=A, X_seq_norm=X_seq_norm, mean_vec=mean_vec, std_vec=std_vec, t0=t,
                steps=steps, node_type=node_type, vel_idxs=vel_idxs, stress_idxs=stress_idxs,
                node_type_idxs=node_type_idxs, world_pos_idxs=world_pos_idxs, add_world_edges=add_world_edges,
                radius=dataconfig.get('radius_world_edge', 0.03))

    # Visualize each step
    for k in range(steps):

        # true values at step k
        X_tp_k_norm = X_seq_norm[t + 1 + k]
        if X_tp_k_norm.dim() == 3:
            X_tp_k_norm = X_tp_k_norm[0]

        coords_true_norm = X_tp_k_norm[:, :3]
        coords_true = coords_true_norm * std_vec[:3] + mean_vec[:3]
        coords_true = coords_true.cpu().numpy()

        # Ground-truth von Mises stress at this step
        stress_true_norm = X_tp_k_norm[:, stress_idxs]
        stress_true = (stress_true_norm * std_vec[stress_idxs] + mean_vec[stress_idxs]).cpu().numpy().squeeze(-1)

        node_type_true = node_type_np

        coords_pred = coords_pred_list[k]
        stress_pred = stress_pred_list[k].squeeze(-1)
        node_type_pred = node_type_pred_list[k]

        coords_true, coords_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells_filtered = apply_render_mode(
            coords_true, coords_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells
        )

        visualize_mesh_pair(pos_true=coords_true, pos_pred=coords_pred, stress_true=stress_true,
                            stress_pred=stress_pred, node_type_true=node_type_true, node_type_pred=node_type_pred,
                            cells=cells_filtered, color_mode="stress", title_true=f"Ground Truth t={t + 1 + k}",
                            title_pred=f"Prediction t={t + 1 + k}", dynamic_edges=dynamic_edges_list[k])

    # Plot rollout error
    print("\nPlotting Rollout Error...")

    fig_err = go.Figure()

    fig_err.add_trace(go.Scatter(x=list(range(1, steps + 1)), y=rollout_error_list, mode='lines+markers',
                                 name='MSE Error', line=dict(color='red', width=2), marker=dict(size=6)))

    fig_err.update_layout(title=f"Rollout Position Error (MSE) over {steps} steps", xaxis_title="Rollout Step",
                          yaxis_title="Mean Squared Error (Physical Units)", template="plotly_white", height=500,
                          width=900, showlegend=True)

    fig_err.show()


if __name__ == "__main__":
    # Visualization settings  [374,356,302,387] overfit_traj_id: 2
    traj_idx = 0
    t_step = 5  # time index t (visualize t -> t+1)
    rollout_set = True  # if True, run multi-step rollout
    rollout_steps = 10  # maximum number of rollout steps for multi-step visualization
    render_mode = "all"  # options: "all", "no_border", "no_sphere", "no_border_no_sphere"
    config_path = os.path.join(os.path.dirname(__file__), "config_gunet.yaml")
    config = load_config(config_path)
    preprocessed_path = config['training']['datapath'] + "/preprocessed_train.pt"

    # Load used_dataconfig.yaml to get data processing parameters
    used_dataconfig_path = os.path.join(config['training']['datapath'], "used_dataconfig.yaml")
    if os.path.exists(used_dataconfig_path):
        dataconfig = load_config(used_dataconfig_path)
    else:
        raise ValueError(f"Warning: {used_dataconfig_path} not found. Using defaults.")
    include_mesh_pos = dataconfig['include_mesh_pos']
    add_world_edges = dataconfig['add_world_edges']

    dataset_name = os.path.basename(os.path.normpath(config['training']["datapath"]))
    out_dir = os.path.join(config['training']["model_path_out"], dataset_name)
    checkpoint_path = os.path.join(out_dir, "model.pt")

    # Use helper to get feature indices
    feat_idx = get_feature_indices(include_mesh_pos)

    mesh_pos_idxs = feat_idx.mesh_pos
    world_pos_idxs = feat_idx.world_pos
    node_type_idxs = feat_idx.nodetype
    vel_idxs = feat_idx.velocity
    stress_idxs = feat_idx.stress
    dim_in = feat_idx.dim_in

    main(mesh_pos_idxs, world_pos_idxs, node_type_idxs, vel_idxs, stress_idxs, dim_in, render_mode, rollout_steps,
         traj_idx, t_step, rollout_set, preprocessed_path, add_world_edges, checkpoint_path)