import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import textwrap
import yaml

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


def plot_feature(save_dir, folder_name, pred, true, label, footer_func):
    """Helper to plot Pred vs True and Residuals for a single feature."""
    pred_dir = os.path.join(save_dir, "pred_vs_true")
    res_dir = os.path.join(save_dir, "residuals")
    os.makedirs(pred_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    # 1. Pred vs True
    plt.figure(figsize=(7, 7))
    if len(true) > 10000:
        idx = np.random.choice(len(true), 10000, replace=False)
        plt.scatter(true[idx], pred[idx], s=6, alpha=0.4)
    else:
        plt.scatter(true, pred, s=8, alpha=0.5)
    
    min_v = min(true.min(), pred.min())
    max_v = max(true.max(), pred.max())
    plt.plot([min_v, max_v], [min_v, max_v], "r--", alpha=0.7)
    
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title(f"Pred vs True - {label}")
    plt.grid(True)
    footer_func()
    plt.savefig(os.path.join(pred_dir, f"{folder_name}_{label.replace(' ', '_')}.png"))
    plt.close()

    # 2. Residuals
    plt.figure(figsize=(8, 5))
    residuals = pred - true
    if len(true) > 10000:
        idx = np.random.choice(len(true), 10000, replace=False)
        plt.scatter(true[idx], residuals[idx], s=6, alpha=0.4)
    else:
        plt.scatter(true, residuals, s=8, alpha=0.5)
    
    plt.axhline(0, color="r", linestyle="--", alpha=0.7)
    plt.xlabel("True")
    plt.ylabel("Residual (Pred-True)")
    plt.title(f"Residuals - {label}")
    plt.grid(True)
    footer_func()
    plt.savefig(os.path.join(res_dir, f"{folder_name}_{label.replace(' ', '_')}.png"))
    plt.close()

    import os
import matplotlib.pyplot as plt
import numpy as np

def make_final_plots(save_dir, train_losses, val_losses, 
                     train_vel_losses=None, val_vel_losses=None, 
                     train_stress_losses=None, val_stress_losses=None, 
                     predictions=None, targets=None,
                     # Handle potential extra args gracefully
                     vel_preds=None, stress_preds=None, 
                     vel_targets=None, stress_targets=None):
    
    os.makedirs(save_dir, exist_ok=True)

    # 1. LOSS CURVES
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.yscale('log')
    plt.legend()
    plt.title('Training Convergence')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()

    # 2. COMPONENT LOSSES (Optional)
    if train_vel_losses is not None:
        plt.figure(figsize=(10, 5))
        plt.plot(train_vel_losses, label='Train Vel', linestyle='--')
        plt.plot(val_vel_losses, label='Val Vel')
        plt.plot(train_stress_losses, label='Train Stress', linestyle='--')
        plt.plot(val_stress_losses, label='Val Stress')
        plt.xlabel('Epoch')
        plt.ylabel('MSE')
        plt.yscale('log')
        plt.legend()
        plt.title('Velocity vs Stress Loss')
        plt.savefig(os.path.join(save_dir, 'component_losses.png'))
        plt.close()

    # 3. PARITY PLOTS
    # Handle different input formats (Combined vs Separated)
    if predictions is not None:
        # Assuming [Vel(3), Stress(1)]
        pred_vel = predictions[:, :3]
        pred_str = predictions[:, 3]
        targ_vel = targets[:, :3]
        targ_str = targets[:, 3]
    elif vel_preds is not None:
        pred_vel = vel_preds
        pred_str = stress_preds.flatten()
        targ_vel = vel_targets
        targ_str = stress_targets.flatten()
    else:
        print("No predictions provided for plotting.")
        return

    # Subsample for speed (max 5000 points)
    if len(pred_vel) > 5000:
        idx = np.random.choice(len(pred_vel), 5000, replace=False)
        pred_vel = pred_vel[idx]
        pred_str = pred_str[idx]
        targ_vel = targ_vel[idx]
        targ_str = targ_str[idx]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Velocity Magnitude
    pred_mag = np.linalg.norm(pred_vel, axis=1)
    targ_mag = np.linalg.norm(targ_vel, axis=1)
    
    axes[0].scatter(targ_mag, pred_mag, alpha=0.3, s=1)
    max_val = max(targ_mag.max(), pred_mag.max())
    axes[0].plot([0, max_val], [0, max_val], 'r--')
    axes[0].set_xlabel('True Speed')
    axes[0].set_ylabel('Pred Speed')
    axes[0].set_title('Velocity Parity')

    # Stress
    axes[1].scatter(targ_str, pred_str, alpha=0.3, s=1)
    min_val = min(targ_str.min(), pred_str.min())
    max_val = max(targ_str.max(), pred_str.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--')
    axes[1].set_xlabel('True Stress')
    axes[1].set_ylabel('Pred Stress')
    axes[1].set_title('Stress Parity')

    plt.savefig(os.path.join(save_dir, 'predictions.png'))
    plt.close()
    print(f"Plots saved to {save_dir}")