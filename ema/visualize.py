import torch
import numpy as np
import yaml
import os

from ema_data_loader_multi import load_all_trajectories
from ema_unet import GNet_EMA
from ema_utils.ema_helper import visualize_mesh_pair


# ----------------------------------------------------------------------
# CONFIGURATION CONSTANTS (EDIT HERE INSTEAD OF CLI ARGS)
# ----------------------------------------------------------------------

# Dataset and trajectory selection
TFRECORD_PATH = "data/train.tfrecord"
META_PATH = "data/meta.json"
TRAJ_INDEX = 0          # which trajectory to visualize

# Model checkpoint
CHECKPOINT_PATH = "gnet_ema_multi.pt"

# Visualization settings
T_STEP = 0              # time index t (visualize t -> t+1)
ROLLOUT = True         # if True, run multi-step rollout
ROLLOUT_STEPS = 10     # maximum number of rollout steps for multi-step visualization

# Wrapper for model args
class ArgsWrapper:
    pass


# ------------------------------------------------------------
# CONFIGURATION LOADING (SHARED WITH TRAINING)
# ------------------------------------------------------------

def load_config(config_path="config.yaml"):
    """Load model and training configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# MULTI-STEP ROLLOUT (USING VELOCITY PREDICTIONS)

def rollout(model, A, X_t_norm, mean, std, steps, node_type):
    """
    Autoregressive rollout that integrates positions from predicted velocities.

    Parameters
    ----------
    model : GNet_EMA
        Current EMA model that predicts [vx, vy, vz, stress] in normalized space.
    A : Tensor [N,N]
        Adjacency matrix.
    X_t_norm : Tensor [N,F]
        Normalized features at initial time t.
    mean, std : Tensors [1,1,F]
        Normalization statistics for this trajectory.
    steps : int
        Number of rollout steps.
    node_type : Tensor [N]
        Integer node types for this mesh (used only for visualization).

    Returns
    -------
    coords_pred_list : list of [N,3] np.arrays
        Predicted world coordinates at each rollout step.
    stress_pred_list : list of [N] np.arrays
        Predicted stress (denormalized) at each rollout step.
    node_type_pred_list : list of [N] np.arrays
        Node types used for visualization (copied from input node_type).
    """

    device = X_t_norm.device
    mean_vec = mean[0, 0].to(device)  # [F]
    std_vec = std[0, 0].to(device)    # [F]
    node_type = node_type.to(device)  # [N]

    coords_pred_list = []
    stress_pred_list = []
    node_type_pred_list = []

    current_norm = X_t_norm  # [N,F]

    for _ in range(steps):

        # 1) De-normalize current state to get physical features
        current_phys = current_norm * std_vec + mean_vec  # [N,F]
        pos_t = current_phys[:, :3]                       # [N,3]

        # 2) Predict next-step normalized velocity + stress
        with torch.no_grad():
            pred = model.rollout_step(A, current_norm)  # [N,4] (vx,vy,vz,stress) normalized

        vel_norm = pred[:, :3]    # [N,3]
        stress_norm = pred[:, 3]  # [N]

        # 3) Denormalize velocity and stress
        vel = vel_norm * std_vec[4:7] + mean_vec[4:7]      # [N,3]
        stress = stress_norm * std_vec[7] + mean_vec[7]    # [N]

        # 4) Integrate position with predicted velocity
        pos_next = pos_t + vel                              # [N,3]

        # 5) Store for visualization
        coords_pred_list.append(pos_next.cpu().numpy())
        stress_pred_list.append(stress.cpu().numpy())
        node_type_pred_list.append(node_type.cpu().numpy())

        # 6) Build next-step physical feature tensor
        X_next_phys = torch.zeros_like(current_phys)
        X_next_phys[:, :3] = pos_next
        X_next_phys[:, 3] = node_type.float()
        X_next_phys[:, 4:7] = vel
        X_next_phys[:, 7] = stress

        # 7) Re-normalize for next input
        current_norm = (X_next_phys - mean_vec) / std_vec

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

    A = traj["A"]                # [N,N]
    X_seq_norm = traj["X_seq_norm"]  # [T,N,F]
    mean = traj["mean"]          # [1,1,F]
    std = traj["std"]            # [1,1,F]
    cells = traj["cells"]        # [C,4]
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
        raise ValueError(f"t must be in [0, {T-2}]")

    X_t_norm = X_seq_norm[t]
    X_tp_norm = X_seq_norm[t+1]

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
    myargs.act_n = model_cfg["act_n"]
    myargs.act_c = model_cfg["act_c"]
    myargs.l_dim = model_cfg["l_dim"]
    myargs.h_dim = model_cfg["h_dim"]
    myargs.ks = model_cfg["ks"]
    myargs.drop_n = model_cfg["drop_n"]
    myargs.drop_c = model_cfg["drop_c"]

    F_in = X_seq_norm.shape[2]
    # Model trained to output [vx,vy,vz,stress]
    model = GNet_EMA(F_in, 4, myargs).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    # ---------------------- ONE-STEP PREDICTION ----------------------
    # Ground-truth coordinates and von Mises stress at t+1 (for comparison)
    mean_vec = mean[0, 0]
    std_vec = std[0, 0]

    coords_true = X_tp_norm[:, :3] * std_vec[:3] + mean_vec[:3]   # [N,3]
    pos_true = coords_true.cpu().numpy()

    stress_true = X_tp_norm[:, 7] * std_vec[7] + mean_vec[7]      # [N] (von Mises stress)
    stress_true = stress_true.cpu().numpy()

    # Use rollout with 1 step to integrate predicted velocities
    coords_pred_list, stress_pred_list, node_type_pred_list = rollout(
        model=model,
        A=A,
        X_t_norm=X_t_norm,
        mean=mean,
        std=std,
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
            color_mode="stress",   # or "node_type"
            title_true=f"Ground Truth t={t+1}",
            title_pred=f"Prediction t={t+1}"
        )
        return

    # ---------------------- MULTI-STEP ROLLOUT ----------------------
    steps = min(ROLLOUT_STEPS, T - 1 - t)
    print(f"\nPerforming {steps}-step rollout...")

    coords_pred_list, stress_pred_list, node_type_pred_list = rollout(
        model=model,
        A=A,
        X_t_norm=X_t_norm,
        mean=mean,
        std=std,
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
        coords_true = coords_true_norm * std[:, :, :3] + mean[:, :, :3]
        coords_true = coords_true.cpu().numpy()

        # Ground-truth von Mises stress at this step
        stress_true_norm = X_tp_k_norm[:, 7]
        stress_true = (stress_true_norm * std_vec[7] + mean_vec[7]).cpu().numpy()

        node_type_true = node_type_np

        coords_pred = coords_pred_list[k]
        stress_pred = stress_pred_list[k]
        node_type_pred = node_type_pred_list[k]

        visualize_mesh_pair(
            pos_true=coords_true,
            pos_pred=coords_pred,
            stress_true=stress_true,
            stress_pred=stress_pred,
            node_type_true=node_type_true,
            node_type_pred=node_type_pred,
            cells=cells,
            color_mode="stress",
            title_true=f"Ground Truth t={t+1+k}",
            title_pred=f"Prediction t={t+1+k}"
        )


if __name__ == "__main__":
    main()
