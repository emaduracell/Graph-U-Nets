import torch
import numpy as np
import yaml
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from data_loader import load_all_trajectories
from graph_unet_defplate import GNet_EMA

TFRECORD_PATH = "data/train.tfrecord"
META_PATH = "data/meta.json"
TRAJ_INDEX = 0
OUTPUT_DIR = "simulation_rollout"
# Model checkpoint
CHECKPOINT_PATH = "gnet_ema_multi.pt"

# Visualization settings
T_STEP = 50  # time index t (visualize t -> t+1)
ROLLOUT = True  # if True, run multi-step rollout
ROLLOUT_STEPS = 400  # maximum number of rollout steps for multi-step visualization


# Wrapper for model args
class ArgsWrapper:
    pass


def load_config(config_path):
    """Load model and training configuration from YAML file, so that it's consistent."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def visualize_mesh_pair(pos_true, pos_pred, cells, stress_true, stress_pred, node_type_true, node_type_pred, title_true,
                        title_pred, color_mode):
    """
    Visualizzazione mesh + heatmap stress o node_type.
    """

    # ======================================================
    # 1) REMOVE BATCH DIMENSION IF PRESENT
    # ======================================================
    if pos_true.ndim == 3:
        pos_true = pos_true[0]
    if pos_pred.ndim == 3:
        pos_pred = pos_pred[0]

    if stress_true is not None and stress_true.ndim == 2:
        stress_true = stress_true[0]
    if stress_pred is not None and stress_pred.ndim == 2:
        stress_pred = stress_pred[0]

    if node_type_true is not None and node_type_true.ndim == 2:
        node_type_true = node_type_true[0]
    if node_type_pred is not None and node_type_pred.ndim == 2:
        node_type_pred = node_type_pred[0]

    # ======================================================
    # 2) TRIANGOLAZIONE CELLE QUADRILATERE
    # ======================================================
    tri_i, tri_j, tri_k = [], [], []
    for (i0, i1, i2, i3) in cells:
        tri_i.extend([i0, i0])
        tri_j.extend([i1, i2])
        tri_k.extend([i2, i3])

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

    # ---------------- TRUE MESH ----------------
    fig.add_trace(
        go.Mesh3d(
            x=pos_true[:, 0], y=pos_true[:, 1], z=pos_true[:, 2],
            i=tri_i, j=tri_j, k=tri_k,
            intensity=intensity_true,
            colorscale=colorscale,
            showscale=True,
            flatshading=True,
            opacity=1.0,
            name="true_mesh"
        ),
        row=1, col=1
    )

    # ---------------- PRED MESH ----------------
    fig.add_trace(go.Mesh3d(x=pos_pred[:, 0], y=pos_pred[:, 1], z=pos_pred[:, 2],
            i=tri_i, j=tri_j, k=tri_k,
            intensity=intensity_pred,
            colorscale=colorscale,
            showscale=True,
            flatshading=True,
            opacity=1.0,
            name="pred_mesh"), row=1, col=2)

    # ======================================================
    # 5) SETTINGS
    # ======================================================
    fig.update_scenes(aspectmode="data")
    fig.update_layout(height=600,
        width=1200,
        title_text="Mesh Comparison",
        showlegend=False)

    fig.show()


# MULTI-STEP ROLLOUT (USING VELOCITY PREDICTIONS)

def rollout(model, A, X_seq_norm, mean, std, t0, steps, node_type):
    """
    Autoregressive rollout that:
      - predicts plate velocities + stresses,
      - keeps borders fixed (node_type == 6),
      - drives rigid body (node_type == 1) with scripted (ground-truth) motion.

    Parameters
    ----------
    model : GNet_EMA
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
    mean_vec = mean[0, 0].to(device)  # [F]
    std_vec = std[0, 0].to(device)  # [F]
    node_type = node_type.to(device)  # [N]

    # Masks
    deform_mask = (node_type == 0)  # deformable plate
    rigid_mask = (node_type == 1)  # rigid body
    border_mask = (node_type == 6)  # fixed borders

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

    for k in range(steps):
        # ======================================================
        # 1) Predict NORMALIZED velocity + stress
        #    v_hat_k, sigma_hat_k from graph at time "k"
        # ======================================================
        with torch.no_grad():
            pred = model.rollout_step(A, current_norm)  # [N,4] normalized

        vel_norm = pred[:, :3]  # [N,3]
        stress_norm = pred[:, 3]  # [N]

        # Denormalize predicted velocity & stress (v_hat_k, sigma_hat_k)
        vel_pred = vel_norm * std_vec[4:7] + mean_vec[4:7]  # [N,3]
        stress_pred = stress_norm * std_vec[7] + mean_vec[7]  # [N]

        # ======================================================
        # 2) p_hat_{k+1} from p_hat_k + v_hat_k (ONLY deformables)
        # ======================================================
        # Start p_hat_{k+1} as a copy of p_hat_k
        p_hat_next = p_hat.clone()  # [N,3]
        stress_next = stress_pred.clone()  # [N]

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
            gt_norm_step = gt_norm_step[0]

        gt_phys_step = gt_norm_step * std_vec + mean_vec  # [N,F]
        p_rigid_gt = gt_phys_step[:, :3]  # [N,3]
        v_rigid_gt = gt_phys_step[:, 4:7]  # [N,3]
        s_rigid_gt = gt_phys_step[:, 7]  # [N]

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

        # ======================================================
        # 6) Build physical features X_{k+1} from (p_hat_{k+1}, v_hat_k+rigid/border overrides)
        # ======================================================
        X_next_phys = torch.zeros_like(current_phys)
        X_next_phys[:, :3] = p_hat_next  # positions
        X_next_phys[:, 3] = node_type.float()  # node type
        X_next_phys[:, 4:7] = vel_pred  # velocity field
        X_next_phys[:, 7] = stress_next  # stress

        # Re-normalize for next model input (graph at time k+1)
        current_phys = X_next_phys
        current_norm = (X_next_phys - mean_vec) / std_vec

        # Advance p_hat_k -> p_hat_{k+1}
        p_hat = p_hat_next

    return coords_pred_list, stress_pred_list, node_type_pred_list


# MAIN VISUALIZATION LOGIC

def main():
    # ---------------------- LOAD DATA ----------------------
    print("Loading trajectory...")
    # Use the same loader used for training
    list_of_trajs = load_all_trajectories(
        TFRECORD_PATH,
        META_PATH,
        max_trajs=TRAJ_INDEX + 1
    )
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
        X_t_norm = X_t_norm[0]
    if X_tp_norm.dim() == 3:
        X_tp_norm = X_tp_norm[0]

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
    model = GNet_EMA(dim_in, 3, 1, myargs).to(device)
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

    stress_true = X_tp_norm[:, 7] * std_vec[7] + mean_vec[7]  # [N] (von Mises stress)
    stress_true = stress_true.cpu().numpy()

    # Use rollout with 1 step to integrate predicted velocities
    coords_pred_list, stress_pred_list, node_type_pred_list = rollout(
        model=model,
        A=A,
        X_seq_norm=X_seq_norm,
        mean=mean,
        std=std,
        t0=t,
        steps=1,
        node_type=node_type,
    )

    pos_pred = coords_pred_list[0]
    stress_pred = stress_pred_list[0]

    # Node types for visualization: always use ground-truth integers
    node_type_np = node_type.cpu().numpy()
    node_type_true = node_type_np
    node_type_pred = node_type_np

    if isinstance(cells, torch.Tensor):
        cells = cells.cpu().numpy()

    # ---------------------- SINGLE STEP VISUALIZE ----------------------
    if not ROLLOUT:
        visualize_mesh_pair(
            pos_true=pos_true,
            pos_pred=pos_pred,
            stress_true=stress_true,
            stress_pred=stress_pred,
            node_type_true=node_type_true,
            node_type_pred=node_type_pred,
            cells=cells,
            color_mode="stress",  # or "node_type"
            title_true=f"Ground Truth t={t + 1}",
            title_pred=f"Prediction t={t + 1}"
        )
        return

    # ---------------------- MULTI-STEP ROLLOUT ----------------------
    steps = min(ROLLOUT_STEPS, T - 1 - t)
    print(f"\nPerforming {steps}-step rollout...")

    coords_pred_list, stress_pred_list, node_type_pred_list = rollout(
        model=model,
        A=A,
        X_seq_norm=X_seq_norm,
        mean=mean,
        std=std,
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
        stress_true_norm = X_tp_k_norm[:, 7]
        stress_true = (stress_true_norm * std_vec[7] + mean_vec[7]).cpu().numpy()

        node_type_true = node_type_np

        coords_pred = coords_pred_list[k]
        stress_pred = stress_pred_list[k]
        node_type_pred = node_type_pred_list[k]

        visualize_mesh_pair(pos_true=coords_true, pos_pred=coords_pred, stress_true=stress_true,
                            stress_pred=stress_pred, node_type_true=node_type_true, node_type_pred=node_type_pred,
                            cells=cells, color_mode="stress", title_true=f"Ground Truth t={t + 1 + k}",
                            title_pred=f"Prediction t={t + 1 + k}")


if __name__ == "__main__":
    main()
