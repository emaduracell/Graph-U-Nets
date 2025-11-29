import argparse
import numpy as np
from ema_utils.ema_helper import visualize_mesh_pair


def load_npy(path):
    arr = np.load(path)
    arr = np.squeeze(arr)
    return arr


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--true_positions", type=str, required=True)
    parser.add_argument("--pred_positions", type=str, required=True)

    parser.add_argument("--true_stress", type=str, default=None)
    parser.add_argument("--pred_stress", type=str, default=None)

    parser.add_argument("--true_velocity", type=str, default=None)
    parser.add_argument("--pred_velocity", type=str, default=None)

    parser.add_argument("--cells", type=str, required=True)

    parser.add_argument("--t", type=int, required=True,
                        help="Time index to visualize")

    parser.add_argument("--color", type=str, default="stress",
                        choices=["stress", "velocity", "none", "position_norm"],
                        help="Color mode for visualization")

    args = parser.parse_args()

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    pos_true_seq = load_npy(args.true_positions)        # [T,N,3]
    pos_pred_seq = load_npy(args.pred_positions)        # [T,N,3]
    cells = load_npy(args.cells).astype(int)            # [C,4]

    # optional stress
    stress_true_seq = load_npy(args.true_stress) if args.true_stress else None
    stress_pred_seq = load_npy(args.pred_stress) if args.pred_stress else None

    # optional velocity
    vel_true_seq = load_npy(args.true_velocity) if args.true_velocity else None
    vel_pred_seq = load_npy(args.pred_velocity) if args.pred_velocity else None

    # -------------------------------------------------
    # SELECT TIME STEP
    # -------------------------------------------------
    t = args.t

    pos_true = pos_true_seq[t]
    pos_pred = pos_pred_seq[t]

    # ---- choose intensity field ----
    if args.color == "stress":
        if stress_true_seq is None or stress_pred_seq is None:
            raise ValueError("Stress files required for color='stress'")

        intensity_true = stress_true_seq[t]
        intensity_pred = stress_pred_seq[t]

    elif args.color == "velocity":
        if vel_true_seq is None or vel_pred_seq is None:
            raise ValueError("Velocity files required for color='velocity'")

        v_true = vel_true_seq[t]
        v_pred = vel_pred_seq[t]

        intensity_true = np.linalg.norm(v_true, axis=1)
        intensity_pred = np.linalg.norm(v_pred, axis=1)

    elif args.color == "position_norm":
        intensity_true = np.linalg.norm(pos_true, axis=1)
        intensity_pred = np.linalg.norm(pos_pred, axis=1)

    else:  # "none"
        intensity_true = None
        intensity_pred = None

    # -------------------------------------------------
    # VISUALIZE (using your existing mesh_pair function)
    # -------------------------------------------------
    visualize_mesh_pair(
        pos_true=pos_true,
        pos_pred=pos_pred,
        stress_true=intensity_true,
        stress_pred=intensity_pred,
        cells=cells,
        color_mode="stress" if args.color != "none" else "stress",
        title_true=f"True (t={t})",
        title_pred=f"Prediction (t={t})",
    )


if __name__ == "__main__":
    main()
