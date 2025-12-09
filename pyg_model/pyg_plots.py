import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import textwrap
import yaml

NORMAL_NODE = 0
BOUNDARY_NODE = 3
VELOCITY_INDEXES = slice(0, 3)  # velocity channels in denormed outputs we pass
STRESS_INDEXES = slice(3, 4)    # stress channel in denormed outputs we pass


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _to_cpu_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        return np.asarray([_to_cpu_numpy(e) for e in x], dtype=object)
    return np.asarray(x)


def _flatten_dict(d, parent_key=""):
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def make_final_plots(save_dir, train_losses, val_losses, train_vel_losses, train_stress_losses, val_vel_losses,
                     val_stress_losses, predictions, targets):
    """
    Generate loss curves and predicted-vs-true / residual plots.
    """
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    flat_cfg = {**_flatten_dict(config.get("model", {}), "model"), **_flatten_dict(config.get("training", {}), "train")}
    footer_text = ", ".join(f"{k}={v}" for k, v in flat_cfg.items())

    def add_cfg_footer():
        if not footer_text:
            return
        wrapped = textwrap.fill(footer_text, width=120)
        plt.subplots_adjust(bottom=0.22)
        plt.figtext(0.5, 0.02, wrapped, ha="center", va="bottom", fontsize=6)

    train_losses = _to_cpu_numpy(train_losses).astype(float)
    val_losses = _to_cpu_numpy(val_losses).astype(float)
    train_vel_losses = _to_cpu_numpy(train_vel_losses).astype(float)
    train_stress_losses = _to_cpu_numpy(train_stress_losses).astype(float)
    val_vel_losses = _to_cpu_numpy(val_vel_losses).astype(float)
    val_stress_losses = _to_cpu_numpy(val_stress_losses).astype(float)

    # Loss curves
    plt.figure(figsize=(8, 5))
    epochs = np.arange(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.grid(True)
    plt.legend()
    add_cfg_footer()
    plt.savefig(os.path.join(save_dir, "loss_vs_epoch.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_vel_losses, label="Train Vel")
    plt.plot(epochs, train_stress_losses, label="Train Stress")
    plt.plot(epochs, val_vel_losses, label="Val Vel")
    plt.plot(epochs, val_stress_losses, label="Val Stress")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Velocity / Stress Loss vs Epoch")
    plt.grid(True)
    plt.legend()
    add_cfg_footer()
    plt.savefig(os.path.join(save_dir, "vel_stress_losses.png"))
    plt.close()

    # Predictions vs targets
    preds = _to_cpu_numpy(predictions)
    targs = _to_cpu_numpy(targets)

    if preds.ndim == 1 or preds.shape[0] == 0:
        return

    feature_labels = ["Vel X", "Vel Y", "Vel Z", "Stress"] if preds.shape[1] == 4 else [f"Feat {i}" for i in range(preds.shape[1])]

    pred_dir = os.path.join(save_dir, "pred_vs_true")
    os.makedirs(pred_dir, exist_ok=True)
    res_dir = os.path.join(save_dir, "residuals")
    os.makedirs(res_dir, exist_ok=True)

    for i in range(preds.shape[1]):
        y_pred = preds[:, i]
        y_true = targs[:, i]

        plt.figure(figsize=(7, 7))
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            plt.scatter(y_true[idx], y_pred[idx], s=6, alpha=0.4)
        else:
            plt.scatter(y_true, y_pred, s=8, alpha=0.5)
        min_v = min(y_true.min(), y_pred.min())
        max_v = max(y_true.max(), y_pred.max())
        plt.plot([min_v, max_v], [min_v, max_v], "r--", alpha=0.7)
        plt.xlabel("True")
        plt.ylabel("Pred")
        plt.title(f"Pred vs True - {feature_labels[i]}")
        plt.grid(True)
        add_cfg_footer()
        plt.savefig(os.path.join(pred_dir, f"feat_{i}.png"))
        plt.close()

        plt.figure(figsize=(8, 5))
        residuals = y_pred - y_true
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            plt.scatter(y_true[idx], residuals[idx], s=6, alpha=0.4)
        else:
            plt.scatter(y_true, residuals, s=8, alpha=0.5)
        plt.axhline(0, color="r", linestyle="--", alpha=0.7)
        plt.xlabel("True")
        plt.ylabel("Residual (Pred-True)")
        plt.title(f"Residuals - {feature_labels[i]}")
        plt.grid(True)
        add_cfg_footer()
        plt.savefig(os.path.join(res_dir, f"feat_{i}.png"))
        plt.close()
