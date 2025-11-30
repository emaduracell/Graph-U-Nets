import torch
import argparse
import numpy as np

from ema_data_loader import data_loader_gnet
from ema_unet import GNet_EMA
from ema_utils.ema_helper import visualize_mesh_pair


# PARSE ARGUMENTS

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--tfrecord", type=str, required=True)
    parser.add_argument("--meta", type=str, required=True)
    parser.add_argument("--traj_index", type=int, default=0)
    parser.add_argument("--checkpoint", type=str, required=True)

    parser.add_argument("--t", type=int, required=True)
    parser.add_argument("--rollout", action="store_true",
                        help="If set, perform 10-step autoregressive rollout.")

    # MUST MATCH TRAINING
    parser.add_argument("--act_n", type=str, default="ELU")
    parser.add_argument("--act_c", type=str, default="ELU")
    parser.add_argument("--l_dim", type=int, default=128)
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--ks", nargs="+", type=float, default=[0.9, 0.8, 0.7])
    parser.add_argument("--drop_n", type=float, default=0.0)
    parser.add_argument("--drop_c", type=float, default=0.0)

    return parser.parse_args()


# Wrapper for model args
class ArgsWrapper:
    pass

# MULTI-STEP ROLLOUT

def rollout(model, A, X_t_norm, mean, std, steps):
    """
    Returns:
        coords_pred_list: list of [N,3]
        stress_pred_list: list of [N]
        node_type_pred_list: list of [N]
    """

    coords_pred_list = []
    stress_pred_list = []
    node_type_pred_list = []

    current_norm = X_t_norm  # [N,F]

    for _ in range(steps):

        # predict next normalized features
        with torch.no_grad():
            pred_norm = model.rollout_step(A, current_norm)  # [N,F]

        # remove batch if needed
        if pred_norm.dim() == 3:
            pred_norm = pred_norm[0]

        # extract & denormalize coordinates
        coords_pred_norm = pred_norm[:, :3]     # [N,3]
        coords_pred = coords_pred_norm * std[:, :, :3] + mean[:, :, :3]
        coords_pred = coords_pred.cpu().numpy()

        # extract & denormalize stress
        stress_pred_norm = pred_norm[:, 7]      # [N]
        stress_pred = stress_pred_norm * std[:, :, 7] + mean[:, :, 7]
        stress_pred = stress_pred.cpu().numpy()

        # extract node_type (categorical)
        node_type_pred = pred_norm[:, 3].cpu().numpy()

        # store
        coords_pred_list.append(coords_pred)
        stress_pred_list.append(stress_pred)
        node_type_pred_list.append(node_type_pred)

        # feed prediction back in normalized space
        current_norm = pred_norm

    return coords_pred_list, stress_pred_list, node_type_pred_list


# MAIN VISUALIZATION LOGIC

def main():
    args = get_args()

    # ---------------------- LOAD DATA ----------------------
    print("Loading trajectory...")
    A, X_seq_norm, mean, std, cells = data_loader_gnet(
        args.tfrecord, args.meta, args.traj_index
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = A.to(device)
    X_seq_norm = X_seq_norm.to(device)
    mean = mean.to(device)
    std = std.to(device)

    # ---------------------- SELECT TIME STEP ----------------------
    T = X_seq_norm.shape[0]
    t = args.t
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
    myargs = ArgsWrapper()
    myargs.act_n = args.act_n
    myargs.act_c = args.act_c
    myargs.l_dim = args.l_dim
    myargs.h_dim = args.h_dim
    myargs.ks = args.ks
    myargs.drop_n = args.drop_n
    myargs.drop_c = args.drop_c

    F_in = X_seq_norm.shape[2]
    model = GNet_EMA(F_in, F_in, myargs).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # ---------------------- ONE-STEP PREDICTION ----------------------
    with torch.no_grad():
        pred_norm = model.rollout_step(A, X_t_norm)

    if pred_norm.dim() == 3:
        pred_norm = pred_norm[0]

    # ---- coordinates ----
    coords_true_norm = X_tp_norm[:, :3]
    coords_pred_norm = pred_norm[:, :3]

    coords_true = coords_true_norm * std[:, :, :3] + mean[:, :, :3]
    coords_pred = coords_pred_norm * std[:, :, :3] + mean[:, :, :3]

    pos_true = coords_true.cpu().numpy()
    pos_pred = coords_pred.cpu().numpy()

    # ---- stress ----
    stress_true_norm = X_tp_norm[:, 7]
    stress_pred_norm = pred_norm[:, 7]

    stress_true = stress_true_norm * std[:, :, 7] + mean[:, :, 7]
    stress_pred = stress_pred_norm * std[:, :, 7] + mean[:, :, 7]

    stress_true = stress_true.cpu().numpy()
    stress_pred = stress_pred.cpu().numpy()

    # ---- node_type ----
    node_type_true = X_tp_norm[:, 3].cpu().numpy()
    node_type_pred = pred_norm[:, 3].cpu().numpy()

    if isinstance(cells, torch.Tensor):
        cells = cells.cpu().numpy()

    # ---------------------- SINGLE STEP VISUALIZE ----------------------
    if not args.rollout:
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
    steps = 10
    print(f"\nPerforming {steps}-step rollout...")

    coords_pred_list, stress_pred_list, node_type_pred_list = rollout(
        model=model,
        A=A,
        X_t_norm=X_t_norm,
        mean=mean,
        std=std,
        steps=steps,
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

        stress_true_norm = X_tp_k_norm[:, 7]
        stress_true = (stress_true_norm * std[:, :, 7] + mean[:, :, 7]).cpu().numpy()

        node_type_true = X_tp_k_norm[:, 3].cpu().numpy()

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
