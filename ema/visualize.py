import torch
import argparse
import numpy as np

from ema_data_loader import data_loader_gnet
from ema_unet import GNet_EMA
from ema_utils.ema_helper import visualize_mesh_pair


# ------------------------------------------------------------
# ARGUMENTS
# ------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecord", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--t", type=int, required=True,
                        help="Visualize prediction for t → t+1.")
    parser.add_argument("--rollout", action="store_true",
                        help="Perform 10-step autoregressive rollout.")

    # MUST MATCH TRAINING
    parser.add_argument("--act_n", type=str, default="ELU")
    parser.add_argument("--act_c", type=str, default="ELU")
    parser.add_argument("--l_dim", type=int, default=128)
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--ks", nargs="+", type=float, default=[0.9, 0.8, 0.7])
    parser.add_argument("--drop_n", type=float, default=0.0)
    parser.add_argument("--drop_c", type=float, default=0.0)

    return parser.parse_args()


class ArgsWrapper:
    pass


# ------------------------------------------------------------
# ROLLOUT (t → t+K)
# ------------------------------------------------------------

def rollout(model, A, X_t_norm, mean, std, steps=10):
    """
    Performs autoregressive rollout with velocity integration.

    Returns:
        coords_list  : list[step] of [N,3] positions
        stress_list  : list[step] of [N]
        node_type_list : list[step] of [N]
    """

    coords_list = []
    stress_list = []
    node_type_list = []

    current_norm = X_t_norm.clone()  # [N,8]

    for _ in range(steps):

        # 1) Predict normalized vel & stress
        vel_pred_norm, stress_pred_norm = model.rollout_step(A, current_norm)
        stress_pred_norm = stress_pred_norm.squeeze(-1)  # [N]

        # 2) Decode node type (constant)
        nt_norm = current_norm[:, 3]
        nt_phys = nt_norm * std[:, :, 3] + mean[:, :, 3]
        node_type = nt_phys.round().long().cpu().detach().numpy()

        # 3) Current pos (physical)
        pos_t_norm = current_norm[:, :3]
        pos_t = pos_t_norm * std[:, :, :3] + mean[:, :, :3]

        # 4) De-normalize velocity
        vel_pred = vel_pred_norm * std[:, :, 4:7] + mean[:, :, 4:7]

        # 5) Integrate position
        pos_pred = pos_t + vel_pred
        coords_list.append(pos_pred.cpu().detach().numpy())

        # 6) De-normalize stress
        stress_pred = stress_pred_norm * std[:, :, 7] + mean[:, :, 7]
        stress_list.append(stress_pred.cpu().detach().numpy())

        node_type_list.append(node_type)

        # 7) Build next normalized state for autoregression
        pos_pred_norm = (pos_pred - mean[:, :, :3]) / std[:, :, :3]
        vel_pred_norm2 = (vel_pred - mean[:, :, 4:7]) / std[:, :, 4:7]
        stress_pred_norm2 = (stress_pred - mean[:, :, 7]) / std[:, :, 7]

        next_state = torch.cat([
            pos_pred_norm,
            nt_norm.unsqueeze(-1),   # ALWAYS normalized node type
            vel_pred_norm2,
            stress_pred_norm2.unsqueeze(-1),
        ], dim=1)

        current_norm = next_state

    return coords_list, stress_list, node_type_list


# ------------------------------------------------------------
# MAIN VISUALIZATION
# ------------------------------------------------------------

def main():
    args = get_args()

    print("Loading trajectory...")
    A, X_seq_norm, mean, std, cells = data_loader_gnet(
        args.tfrecord, args.meta, args.traj_index
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)
    X_seq_norm = X_seq_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)

    # ---------------------- TIME SELECTION ----------------------
    T = X_seq_norm.shape[0]
    t = args.t
    if not (0 <= t < T - 1):
        raise ValueError(f"t must be in [0, {T-2}]")

    X_t_norm = X_seq_norm[t]
    X_tp_norm = X_seq_norm[t + 1]

    if X_t_norm.dim() == 3:  X_t_norm = X_t_norm[0]
    if X_tp_norm.dim() == 3: X_tp_norm = X_tp_norm[0]

    # ---------------------- LOAD MODEL ----------------------
    myargs = ArgsWrapper()
    myargs.act_n = args.act_n
    myargs.act_c = args.act_c
    myargs.l_dim = args.l_dim
    myargs.h_dim = args.h_dim
    myargs.ks = args.ks
    myargs.drop_n = args.drop_n
    myargs.drop_c = args.drop_c

    F_in = X_seq_norm.shape[2]  # normally 8
    model = GNet_EMA(F_in, myargs).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # ---------------------- PREDICT SINGLE STEP ----------------------
    with torch.no_grad():
        vel_pred_norm, stress_pred_norm = model.rollout_step(A, X_t_norm)

    stress_pred_norm = stress_pred_norm.squeeze(-1)

    # ---- de-normalize position ----
    pos_t = X_t_norm[:, :3] * std[:, :, :3] + mean[:, :, :3]
    pos_tp = X_tp_norm[:, :3] * std[:, :, :3] + mean[:, :, :3]

    # ---- de-normalize velocity ----
    vel_pred = vel_pred_norm * std[:, :, 4:7] + mean[:, :, 4:7]

    # ---- integrate ----
    pos_pred = pos_t + vel_pred

    # ---- de-normalize stress ----
    stress_tp = X_tp_norm[:, 7] * std[:, :, 7] + mean[:, :, 7]
    stress_pred = stress_pred_norm * std[:, :, 7] + mean[:, :, 7]

    # ---- node types ----
    node_type = (X_t_norm[:, 3] * std[:, :, 3] + mean[:, :, 3]).round().long().cpu().numpy()

    if isinstance(cells, torch.Tensor):
        cells = cells.cpu().numpy()

    pos_true = pos_tp.cpu().numpy()
    pos_pred = pos_pred.cpu().numpy()
    stress_true = stress_tp.cpu().numpy()
    stress_pred = stress_pred.cpu().numpy()

    # ---------------------- SINGLE STEP VISUALIZATION ----------------------
    if not args.rollout:
        visualize_mesh_pair(
            pos_true=pos_true,
            pos_pred=pos_pred,
            stress_true=stress_true,
            stress_pred=stress_pred,
            node_type_true=node_type,
            node_type_pred=node_type,
            cells=cells,
            color_mode="stress",
            title_true=f"Ground Truth t={t+1}",
            title_pred=f"Prediction t={t+1}"
        )
        return

    # ---------------------- MULTI-STEP ROLLOUT ----------------------
    print("Performing 10-step rollout...\n")
    coords_list, stress_list, node_type_list = rollout(
        model=model,
        A=A,
        X_t_norm=X_t_norm,
        mean=mean,
        std=std,
        steps=10
    )

    for k in range(10):

        X_tp_k_norm = X_seq_norm[t + 1 + k]
        if X_tp_k_norm.dim() == 3:
            X_tp_k_norm = X_tp_k_norm[0]

        coords_true = (
            X_tp_k_norm[:, :3] * std[:, :, :3] + mean[:, :, :3]
        ).cpu().detach().numpy()

        stress_true_k = (
            X_tp_k_norm[:, 7] * std[:, :, 7] + mean[:, :, 7]
        ).cpu().detach().numpy()
        node_type_k = (
            (X_tp_k_norm[:, 3] * std[:, :, 3] + mean[:, :, 3])
            .round().long().cpu().detach().numpy()
        )

        visualize_mesh_pair(
            pos_true=coords_true,
            pos_pred=coords_list[k],
            stress_true=stress_true_k,
            stress_pred=stress_list[k],
            node_type_true=node_type_k,
            node_type_pred=node_type_list[k],
            cells=cells,
            color_mode="stress",
            title_true=f"Ground Truth t={t+1+k}",
            title_pred=f"Prediction t={t+1+k}"
        )


if __name__ == "__main__":
    main()
