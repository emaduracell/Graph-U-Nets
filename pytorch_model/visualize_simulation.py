import torch
import numpy as np
import yaml
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from defplate_dataset import add_edges
from model_entire import GraphUNet_DefPlate

# Data paths
TFRECORD_PATH = "data/train.tfrecord"
META_PATH = "data/meta.json"
# Prefer the preprocessed dataset to mirror training
PREPROCESSED_PATH = "data/preprocessed_train.pt"
OUTPUT_DIR = "simulation_rollout"
CHECKPOINT_PATH = "gnet_ema_multi.pt"
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1
NODE_TYPE_INDEXES = slice(3, 5)
VELOCITY_INDEXES = slice(5, 8)  # like 5:8
STRESS_INDEXES = slice(8, 9)  # like 8:9

# Visualization settings  [374,356,302,387] overfit_traj_id: 2
TRAJ_INDEX = 0
T_STEP = 0  # time index t (visualize t -> t+1)
ROLLOUT = True  # if True, run multi-step rollout
ROLLOUT_STEPS = 100  # maximum number of rollout steps for multi-step visualization
RENDER_MODE = "all"  # options: "all", "no_border", "no_sphere", "no_border_no_sphere"


def make_dynamic_edges_trace(coords, edge_index):
    """
    Creates Red Lines for the dynamic interactions between Sphere and Plate.
    """
    if edge_index is None or edge_index.shape[1] == 0:
        print("[visualize_simulation] empty trace")
        return go.Scatter3d()  # Empty trace

    src = edge_index[0].cpu().numpy()
    dst = edge_index[1].cpu().numpy()

    x_lines, y_lines, z_lines = [], [], []
    for s, d in zip(src, dst):
        x_lines.extend([coords[s, 0], coords[d, 0], None])
        y_lines.extend([coords[s, 1], coords[d, 1], None])
        z_lines.extend([coords[s, 2], coords[d, 2], None])

    return go.Scatter3d(x=x_lines, y=y_lines, z=z_lines, mode='lines',
        line=dict(color='red', width=4),  # Thick red lines
        name='Interaction')


def make_wireframe(x, y, z, i, j, k, color='black', width=1.5):
    """
    Creates a Scatter3d trace that draws the edges of the triangles.
    """
    # 1. Create a list of lines: (i->j), (j->k), (k->i)
    # To do this efficiently in Plotly without loops, we stack coordinates
    # in the order: point_i, point_j, point_k, point_i, None
    # 'None' breaks the line so we don't connect separate triangles.

    tri_points = np.vstack([
        i, j, k, i,
        np.full_like(i, -1)  # Placeholder for None
    ]).T.flatten()

    # Map indices to coordinates
    # We replace the -1 indices with NaN or None effectively by masking later
    xe = x[tri_points]
    ye = y[tri_points]
    ze = z[tri_points]

    # Insert None where we had the -1 index to break the lines
    # (The standard Plotly trick for disconnected lines)
    xe[4::5] = None
    ye[4::5] = None
    ze[4::5] = None

    return go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode='lines',
        line=dict(color=color, width=width),
        name='wireframe',
        showlegend=False,
        hoverinfo='skip'  # Don't show tooltips for the grid lines
    )


# Wrapper for model args
class ArgsWrapper:
    pass


def load_config(config_path):
    """Load model and training configuration from YAML file, so that it's consistent."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def visualize_mesh_pair(pos_true, pos_pred, cells, stress_true, stress_pred, node_type_true, node_type_pred, title_true,
                        title_pred, color_mode, dynamic_edges=None):
    """
    Visualizzazione mesh + heatmap stress o node_type.
    """

    # ======================================================
    # 1) REMOVE BATCH DIMENSION IF PRESENT
    # ======================================================
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

    # ======================================================
    # 2) TRIANGOLAZIONE CELLE QUADRILATERE
    # ======================================================
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

    # ======================================================
    # 3) COLORI / HEATMAP
    # ======================================================

    # --- STRESS MODE ---
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

    # --- NODE TYPE MODE ---
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

    # ======================================================
    # 4) FIGURE SETUP
    # ======================================================
    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=(title_true, title_pred)
    )

    # ---------------- TRUE MESH (Surface + Wireframe) ----------------
    # 1. The Surface
    fig.add_trace(
        go.Mesh3d(
            x=pos_true[:, 0], y=pos_true[:, 1], z=pos_true[:, 2],
            i=tri_i, j=tri_j, k=tri_k,
            intensity=intensity_true,
            colorscale=colorscale,
            showscale=True,
            flatshading=True,
            opacity=0.85,  # Slight transparency helps see the grid better
            name="true_mesh"
        ),
        row=1, col=1
    )
    # 2. The Wireframe
    fig.add_trace(
        make_wireframe(pos_true[:, 0], pos_true[:, 1], pos_true[:, 2],
                       np.array(tri_i), np.array(tri_j), np.array(tri_k)),
        row=1, col=1
    )

    # ---------------- PRED MESH (Surface + Wireframe) ----------------
    # 1. The Surface
    fig.add_trace(
        go.Mesh3d(
            x=pos_pred[:, 0], y=pos_pred[:, 1], z=pos_pred[:, 2],
            i=tri_i, j=tri_j, k=tri_k,
            intensity=intensity_pred,
            colorscale=colorscale,
            showscale=True,
            flatshading=True,
            opacity=0.85,
            name="pred_mesh"
        ),
        row=1, col=2
    )
    # 2. The Wireframe
    fig.add_trace(make_wireframe(pos_pred[:, 0], pos_pred[:, 1], pos_pred[:, 2],
                                 np.array(tri_i), np.array(tri_j), np.array(tri_k)), row=1, col=2)
    print(f"dynamic_edges={dynamic_edges}")
    if dynamic_edges is not None:
        fig.add_trace(make_dynamic_edges_trace(pos_pred, dynamic_edges), row=1, col=2)
    # ======================================================
    # 5) SETTINGS
    # ======================================================
    fig.update_scenes(aspectmode="data")
    fig.update_layout(height=600, width=1200, title_text="Mesh Comparison")
    fig.show()


def apply_render_mode(pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells):
    """
    Optionally drop border and/or sphere nodes before rendering based on RENDER_MODE.

    Returns filtered copies of inputs with remapped cell indices.
    """
    mode = RENDER_MODE.lower()
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


# MULTI-STEP ROLLOUT (USING VELOCITY PREDICTIONS)

def rollout(model, A, X_seq_norm, mean_vec, std_vec, t0, steps, node_type):
    """
    Autoregressive rollout that:
      - predicts plate velocities + stresses,
      - keeps borders fixed (node_type == 3),
      - drives rigid body (node_type == 1) with scripted (ground-truth) motion.

    Parameters
    ----------
    model : GraphUNet_DefPlate
    A : Tensor [N,N]
        Adjacency matrix.
    X_seq_norm : Tensor [T,N,F]
        Normalized feature sequence for this trajectory.
    mean, std : Tensors [1,1,F]
        Normalization stats.
    t0 : int
        Starting time index.
    steps : int
        Number of rollout steps.
    node_type : Tensor [N]
        Integer node types.

    Returns
    -------
    coords_pred_list : list of [N,3] np.arrays
    stress_pred_list : list of [N]   np.arrays
    node_type_pred_list : list of [N] np.arrays
    """

    device = A.device
    mean_vec = mean_vec.to(device)  # [F]
    std_vec = std_vec.to(device)  # [F]
    node_type = node_type.to(device)  # [N]
    # Node type features must follow data_loader encoding: [sphere, boundary]
    node_type_onehot = torch.zeros((node_type.shape[0], 2), device=device, dtype=mean_vec.dtype)
    node_type_onehot[:, 0] = (node_type == SPHERE_NODE).float()
    node_type_onehot[:, 1] = (node_type == BOUNDARY_NODE).float()

    # Masks
    deform_mask = (node_type == NORMAL_NODE)  # deformable plate
    rigid_mask = (node_type == SPHERE_NODE)  # rigid body
    border_mask = (node_type == BOUNDARY_NODE)  # fixed borders

    # ---------- initial state at t0 ----------
    current_norm = X_seq_norm[t0].to(device)  # [N,F] or [1,N,F]
    if current_norm.dim() == 3:
        current_norm = current_norm[0]

    current_phys = current_norm * std_vec + mean_vec  # [N,F]
    # This is p_hat_0 := p_0 (ground truth at t0)
    p_hat = current_phys[:, :3].clone()  # [N,3]

    # Borders reference positions (fixed in time)
    pos_border_ref = p_hat[border_mask].clone()  # [Nb,3]

    coords_pred_list = []
    stress_pred_list = []
    node_type_pred_list = []
    rollout_error_list = []
    base_A = A.clone() # Keep the static mesh safe
    dynamic_edges_list = [] # Store edges for viz
    for k in range(steps):
        A_dynamic, dyn_edges = add_edges(base_A, node_type, p_hat, k=5)
        dynamic_edges_list.append(dyn_edges)
        # ======================================================
        # 1) Predict NORMALIZED velocity + stress
        #    v_hat_k, sigma_hat_k from graph at time "k"
        # ======================================================
        with torch.no_grad():
            pred = model.rollout_step(A_dynamic, current_norm)  # [N,4] normalized

        vel_norm = pred[:, :3]  # [N,3]
        stress_norm = pred[:, 3].unsqueeze(-1)  # [N,1]

        # Denormalize predicted velocity & stress (v_hat_k, sigma_hat_k)
        vel_pred = vel_norm * std_vec[VELOCITY_INDEXES] + mean_vec[VELOCITY_INDEXES]  # [N,3]
        stress_pred = stress_norm * std_vec[STRESS_INDEXES] + mean_vec[STRESS_INDEXES]  # [N,1]

        # ======================================================
        # 2) p_hat_{k+1} from p_hat_k + v_hat_k (ONLY deformables)
        # ======================================================
        # Start p_hat_{k+1} as a copy of p_hat_k
        p_hat_next = p_hat.clone()  # [N,3]
        stress_next = stress_pred.clone()  # [N,1]

        # --- deformable plate nodes (node_type == 0) ---
        # This line enforces the desired recurrence strictly:
        #   p_hat_{k+1} = p_hat_k + v_hat_k  (for deformable nodes)
        p_hat_next[deform_mask] = p_hat[deform_mask] + vel_pred[deform_mask]

        # ======================================================
        # 3) Rigid body nodes: follow scripted (ground-truth) motion
        #    at time t0 + 1 + k
        # ======================================================
        gt_norm_step = X_seq_norm[t0 + 1 + k].to(device)  # [N,F] or [1,N,F]
        if gt_norm_step.dim() == 3:
            raise ValueError("gt_norm_step should not have a batch dimension")

        gt_phys_step = gt_norm_step * std_vec + mean_vec  # [N,F]
        p_rigid_gt = gt_phys_step[:, :3]  # [N,3]
        v_rigid_gt = gt_phys_step[:, VELOCITY_INDEXES]  # [N,3]
        s_rigid_gt = gt_phys_step[:, STRESS_INDEXES]  # [N,1]

        # drive rigid nodes with GT
        p_hat_next[rigid_mask] = p_rigid_gt[rigid_mask]
        vel_pred[rigid_mask] = v_rigid_gt[rigid_mask]
        stress_next[rigid_mask] = s_rigid_gt[rigid_mask]

        # ======================================================
        # 4) Fixed borders: fixed positions + zero velocity
        # ======================================================
        p_hat_next[border_mask] = pos_border_ref
        vel_pred[border_mask] = 0.0  # explicitly zero velocity
        # stress_next[border_mask] stays as model prediction

        # ======================================================
        # 5) Store p_hat_{k+1} for visualization
        # ======================================================
        coords_pred_list.append(p_hat_next.detach().cpu().numpy())
        stress_pred_list.append(stress_next.detach().cpu().numpy())
        node_type_pred_list.append(node_type.detach().cpu().numpy())
        p_gt_next = gt_phys_step[:, :3]
        # Calculate MSE: mean((Pred - True)^2)
        mse_step = torch.mean((p_hat_next - p_gt_next) ** 2)
        rollout_error_list.append(mse_step.item())
        # ======================================================
        # 6) Build physical features X_{k+1} from (p_hat_{k+1}, v_hat_k+rigid/border overrides)
        # ======================================================
        X_next_phys = torch.zeros_like(current_phys)
        X_next_phys[:, :3] = p_hat_next  # positions
        X_next_phys[:, NODE_TYPE_INDEXES] = node_type_onehot  # node type one-hot
        X_next_phys[:, VELOCITY_INDEXES] = vel_pred  # velocity field
        X_next_phys[:, STRESS_INDEXES] = stress_next  # stress

        # Re-normalize for next model input (graph at time k+1)
        current_phys = X_next_phys
        current_norm = (X_next_phys - mean_vec) / std_vec

        # Advance p_hat_k -> p_hat_{k+1}
        p_hat = p_hat_next

    return coords_pred_list, stress_pred_list, node_type_pred_list, rollout_error_list, dynamic_edges_list


# MAIN VISUALIZATION LOGIC

def main():
    # ---------------------- LOAD DATA ----------------------
    print("Loading trajectory...")
    # Prefer preprocessed data (same as training); fall back to raw TFRecord if missing
    if os.path.exists(PREPROCESSED_PATH):
        list_of_trajs = torch.load(PREPROCESSED_PATH)
        if TRAJ_INDEX >= len(list_of_trajs):
            raise ValueError(f"Requested TRAJ_INDEX={TRAJ_INDEX} but only {len(list_of_trajs)} trajectories in "
                             f"{PREPROCESSED_PATH}")
        # Keep only what we need to avoid extra host->device transfers
        list_of_trajs = list_of_trajs[:TRAJ_INDEX + 1]
        print(f"Loaded {len(list_of_trajs)} preprocessed trajectories from {PREPROCESSED_PATH}")
    else:
        raise ValueError(f"Preprocessed data not found at {PREPROCESSED_PATH}")
    traj = list_of_trajs[TRAJ_INDEX]

    A = traj["A"]  # [N,N]
    X_seq_norm = traj["X_seq_norm"]  # [T,N,F]
    mean = traj["mean"]  # [1,1,F]
    std = traj["std"]  # [1,1,F]
    cells = traj["cells"]  # [C,4]
    node_type = traj["node_type"]  # [N]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)
    X_seq_norm = X_seq_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)
    node_type = node_type.to(device)

    # ---------------------- SELECT TIME STEP ----------------------
    T = X_seq_norm.shape[0]
    t = T_STEP
    if not (0 <= t < T - 1):
        raise ValueError(f"t must be in [0, {T - 2}]")

    X_t_norm = X_seq_norm[t]
    X_tp_norm = X_seq_norm[t + 1]

    # remove batch dim if needed
    if X_t_norm.dim() == 3:
        raise ValueError("X_t_norm should not have a batch dimension")
    if X_tp_norm.dim() == 3:
        raise ValueError("X_tp_norm should not have a batch dimension")

    # ---------------------- BUILD MODEL ----------------------
    # Load model hyperparameters from the same YAML used in training
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
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

    dim_in = X_seq_norm.shape[2]
    # Model trained to output [vx,vy,vz,stress]
    model = GraphUNet_DefPlate(dim_in, 3, 1, myargs).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)

    # Backwards compatibility: old checkpoints used "s_gcn" instead of "start_gcn"
    if "s_gcn.proj.weight" in state:
        state["start_gcn.proj.weight"] = state.pop("s_gcn.proj.weight")
    if "s_gcn.proj.bias" in state:
        state["start_gcn.proj.bias"] = state.pop("s_gcn.proj.bias")
    model.load_state_dict(state)
    model.eval()

    # ---------------------- ONE-STEP PREDICTION ----------------------
    # Ground-truth coordinates and von Mises stress at t+1 (for comparison)
    mean_vec = mean[0, 0]
    std_vec = std[0, 0]

    coords_true = X_tp_norm[:, :3] * std_vec[:3] + mean_vec[:3]  # [N,3]
    pos_true = coords_true.cpu().numpy()

    stress_true = X_tp_norm[:, STRESS_INDEXES] * std_vec[STRESS_INDEXES] + mean_vec[
        STRESS_INDEXES]  # [N,1] (von Mises stress)
    stress_true = stress_true.cpu().numpy().squeeze(-1)

    # Use rollout with 1 step to integrate predicted velocities
    coords_pred_list, stress_pred_list, node_type_pred_list, rollout_error_list, dynamic_edges_list = rollout(
        model=model,
        A=A,
        X_seq_norm=X_seq_norm,
        mean_vec=mean_vec,
        std_vec=std_vec,
        t0=t,
        steps=1,
        node_type=node_type,
    )

    pos_pred = coords_pred_list[0]
    stress_pred = stress_pred_list[0].squeeze(-1)

    # Node types for visualization: always use ground-truth integers
    node_type_np = node_type.cpu().numpy()
    node_type_true = node_type_np
    node_type_pred = node_type_np

    if isinstance(cells, torch.Tensor):
        cells = cells.cpu().numpy()

    # ---------------------- SINGLE STEP VISUALIZE ----------------------
    pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells_filtered = apply_render_mode(
        pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells
    )

    if not ROLLOUT:
        visualize_mesh_pair(
            pos_true=pos_true,
            pos_pred=pos_pred,
            stress_true=stress_true,
            stress_pred=stress_pred,
            node_type_true=node_type_true,
            node_type_pred=node_type_pred,
            cells=cells_filtered,
            color_mode="stress",  # or "node_type"
            title_true=f"Ground Truth t={t + 1}",
            title_pred=f"Prediction t={t + 1}"
        )
        return

    # ---------------------- MULTI-STEP ROLLOUT ----------------------
    steps = min(ROLLOUT_STEPS, T - 1 - t)
    print(f"\nPerforming {steps}-step rollout...")

    coords_pred_list, stress_pred_list, node_type_pred_list, rollout_error_list, dynamic_edges_list = rollout(
        model=model,
        A=A,
        X_seq_norm=X_seq_norm,
        mean_vec=mean_vec,
        std_vec=std_vec,
        t0=t,
        steps=steps,
        node_type=node_type,
    )

    # ---- visualize each step ----
    for k in range(steps):

        # true values at step k
        X_tp_k_norm = X_seq_norm[t + 1 + k]
        if X_tp_k_norm.dim() == 3:
            X_tp_k_norm = X_tp_k_norm[0]

        coords_true_norm = X_tp_k_norm[:, :3]
        coords_true = coords_true_norm * std_vec[:3] + mean_vec[:3]
        coords_true = coords_true.cpu().numpy()

        # Ground-truth von Mises stress at this step
        stress_true_norm = X_tp_k_norm[:, STRESS_INDEXES]
        stress_true = (stress_true_norm * std_vec[STRESS_INDEXES] + mean_vec[STRESS_INDEXES]).cpu().numpy().squeeze(-1)

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

    # ======================================================
    # PLOT ROLLOUT ERROR
    # ======================================================
    print("\nPlotting Rollout Error...")

    fig_err = go.Figure()

    fig_err.add_trace(go.Scatter(
        x=list(range(1, steps + 1)),
        y=rollout_error_list,
        mode='lines+markers',
        name='MSE Error',
        line=dict(color='red', width=2),
        marker=dict(size=6)
    ))

    fig_err.update_layout(
        title=f"Rollout Position Error (MSE) over {steps} steps",
        xaxis_title="Rollout Step",
        yaxis_title="Mean Squared Error (Physical Units)",
        template="plotly_white",
        height=500,
        width=900,
        showlegend=True
    )

    fig_err.show()


if __name__ == "__main__":
    main()