import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import textwrap
import yaml

# NORMAL_NODE = 0
# SPHERE_NODE = 1
# BOUNDARY_NODE = 3


def load_config(config_path):
    """Load model and training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def _flatten_dict(d, parent_key=""):
    """Helper: flatten config dicts."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}.{k}" if parent_key else str(k)
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _add_cfg_footer(model_cfg, train_cfg):
    """Add a small text block with model_cfg and train_cfg at the bottom."""
    flat_model = _flatten_dict(model_cfg, "model")
    flat_train = _flatten_dict(train_cfg, "train")
    footer_items = list(flat_model.items()) + list(flat_train.items())
    footer_text = ", ".join(f"{k}={v}" for k, v in footer_items)

    if not footer_text:
        return
    wrapped = textwrap.fill(footer_text, width=150)
    plt.subplots_adjust(bottom=0.25)
    plt.figtext(0.5, 0.01, wrapped, ha='center', va='bottom', fontsize=6)


def to_cpu_numpy(x):
    """Convert Torch tensors/lists to NumPy array on CPU."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    if isinstance(x, (list, tuple)):
        if len(x) == 0: return np.array([])
        if isinstance(x[0], torch.Tensor):
            out = []
            for t in x:
                t = t.detach().cpu()
                if t.dim() == 0:
                    out.append(t.item())
                else:
                    out.append(t.numpy())
            return np.array(out, dtype=object if any(isinstance(e, np.ndarray) for e in out) else float)
        return np.asarray(x)
    return np.asarray(x)


# ------------------------------------------------------------------------------
# INDIVIDUAL PLOTTING FUNCTIONS
# ------------------------------------------------------------------------------

def plot_loss_curves(save_dir, epochs, train_losses, val_losses, train_metrics, val_metrics, metric_name, model_cfg,
                     train_cfg):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    if train_metrics is not None and val_metrics is not None:
        plt.plot(epochs, train_metrics, label=f'Train {metric_name}', linestyle='--')
        plt.plot(epochs, val_metrics, label=f'Val {metric_name}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Training/Validation Loss and Metric vs Epoch')
    plt.legend()
    plt.grid(True)
    _add_cfg_footer(model_cfg, train_cfg)
    plt.savefig(os.path.join(save_dir, 'loss_metric_vs_epoch.png'))
    plt.close()


def plot_component_losses(save_dir, epochs, train_stress, train_vel, test_stress, test_vel, model_cfg, train_cfg):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_stress, label='Train stress loss')
    plt.plot(epochs, train_vel, label='Train velocity loss')
    plt.plot(epochs, test_stress, label='Test stress loss')
    plt.plot(epochs, test_vel, label='Test velocity loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.title('Stress and velocity losses')
    plt.legend()
    plt.grid(True)
    _add_cfg_footer(model_cfg, train_cfg)
    plt.savefig(os.path.join(save_dir, 'stress_vel_losses.png'))
    plt.close()


def plot_grad_norms(save_dir, epochs, grad_norms, model_cfg, train_cfg):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, grad_norms, label='Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.title('Average Gradient Norm per Epoch')
    plt.legend()
    plt.grid(True)
    _add_cfg_footer(model_cfg, train_cfg)
    plt.savefig(os.path.join(save_dir, 'grad_norm_vs_epoch.png'))
    plt.close()


def plot_weights(save_dir, model, model_cfg, train_cfg):
    weights_dir = os.path.join(save_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.figure(figsize=(8, 5))
            data = param.data.cpu().numpy().flatten()
            plt.hist(data, bins=50, alpha=0.7)
            plt.title(f'Weight Distribution - {name}')
            plt.grid(True)
            _add_cfg_footer(model_cfg, train_cfg)
            clean_name = name.replace('.', '_')
            plt.savefig(os.path.join(weights_dir, f'{clean_name}.png'))
            plt.close()


def plot_activations(save_dir, activations, model_cfg, train_cfg):
    acts_dir = os.path.join(save_dir, 'activations')
    os.makedirs(acts_dir, exist_ok=True)
    for name, acts in activations.items():
        plt.figure(figsize=(8, 5))
        plt.hist(acts.flatten(), bins=50, alpha=0.7)
        plt.title(f'Activation Distribution - {name}')
        plt.grid(True)
        _add_cfg_footer(model_cfg, train_cfg)
        clean_name = name.replace('.', '_')
        plt.savefig(os.path.join(acts_dir, f'{clean_name}.png'))
        plt.close()


def plot_pred_vs_true(save_dir, feature_labels, iter_targets, iter_predictions, model_cfg, train_cfg,
                      folder_name="pred_vs_true", suffix=""):
    """
    Generic Scatter plot function.
    """
    pred_dir = os.path.join(save_dir, folder_name)
    os.makedirs(pred_dir, exist_ok=True)
    for i, label in enumerate(feature_labels):
        plt.figure(figsize=(8, 8))
        y_true = iter_targets[i]
        y_pred = iter_predictions[i]

        # Subsampling for speed/clarity
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            y_true_plot = y_true[idx]
            y_pred_plot = y_pred[idx]
            s_size = 5
        else:
            y_true_plot = y_true
            y_pred_plot = y_pred
            s_size = 10

        plt.scatter(y_true_plot, y_pred_plot, alpha=0.3, s=s_size)

        # Diagonal line
        if len(y_true) > 0:
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

        plt.xlabel(f'True Value {suffix}')
        plt.ylabel(f'Predicted Value {suffix}')
        plt.title(f'Predicted vs True {suffix} - {label}')
        plt.grid(True)
        _add_cfg_footer(model_cfg, train_cfg)
        plt.savefig(os.path.join(pred_dir, f'feat_{i}.png'))
        plt.close()


def plot_residuals(save_dir, feature_labels, iter_targets, iter_predictions, model_cfg, train_cfg,
                   folder_name="residuals", suffix=""):
    """
    Generic Residual plot function.
    """
    res_dir = os.path.join(save_dir, folder_name)
    os.makedirs(res_dir, exist_ok=True)
    for i, label in enumerate(feature_labels):
        plt.figure(figsize=(10, 6))
        y_true = iter_targets[i]
        y_pred = iter_predictions[i]
        res = y_pred - y_true

        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            y_true_plot = y_true[idx]
            res_plot = res[idx]
            s_size = 5
        else:
            y_true_plot = y_true
            res_plot = res
            s_size = 10

        plt.scatter(y_true_plot, res_plot, alpha=0.3, s=s_size)
        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel(f'True Value {suffix}')
        plt.ylabel(f'Residual (Pred - True) {suffix}')
        plt.title(f'Residuals vs True {suffix} - {label}')
        plt.grid(True)
        _add_cfg_footer(model_cfg, train_cfg)
        plt.savefig(os.path.join(res_dir, f'feat_{i}.png'))
        plt.close()


# ------------------------------------------------------------------------------
# MAIN PLOTTING FUNCTION
# ------------------------------------------------------------------------------

def make_final_plots(save_dir, train_losses, val_losses, metric_name, train_metrics, val_metrics, grad_norms, model,
                     activations,
                     predictions, targets,  # Denormalized
                     predictions_norm, targets_norm,  # Normalized
                     train_vel_losses, train_stress_losses, test_vel_losses,
                     test_stress_losses, velocity_idxs, stress_idxs):
    # 1. Prepare Data (Convert to CPU Numpy)
    train_losses = to_cpu_numpy(train_losses)
    val_losses = to_cpu_numpy(val_losses)
    train_metrics = to_cpu_numpy(train_metrics) if train_metrics is not None else None
    val_metrics = to_cpu_numpy(val_metrics) if val_metrics is not None else None
    grad_norms = to_cpu_numpy(grad_norms)
    train_stress = to_cpu_numpy(train_stress_losses)
    train_vel = to_cpu_numpy(train_vel_losses)
    test_stress = to_cpu_numpy(test_stress_losses)
    test_vel = to_cpu_numpy(test_vel_losses)

    # Helper to unpack lists of arrays
    def prepare_lists(preds, tgts):
        if isinstance(preds, list):
            iter_p = [to_cpu_numpy(p) for p in preds]
            iter_t = [to_cpu_numpy(t) for t in tgts]
            return iter_p, iter_t
        else:
            p = to_cpu_numpy(preds)
            t = to_cpu_numpy(tgts)
            # Default fallback slicing if raw tensors passed
            iter_p = [p[:, i] for i in range(p.shape[1])]
            iter_t = [t[:, i] for i in range(t.shape[1])]
            return iter_p, iter_t

    # Prepare Denormalized
    iter_predictions, iter_targets = prepare_lists(predictions, targets)
    # Prepare Normalized
    iter_predictions_norm, iter_targets_norm = prepare_lists(predictions_norm, targets_norm)

    # Labels
    num_features = len(iter_predictions)
    feature_labels = ['Vel X', 'Vel Y', 'Vel Z', 'Stress'] if num_features == 4 else [f'Feat {i}' for i in
                                                                                      range(num_features)]

    # 2. Load Config for Footer
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    model_cfg = config['model']
    train_cfg = config['training']
    epochs = range(1, len(train_losses) + 1)

    # 3. Call Plotting Functions
    print("Plotting Loss Curves...")
    plot_loss_curves(save_dir, epochs, train_losses, val_losses, train_metrics, val_metrics, metric_name, model_cfg,
                     train_cfg)

    print("Plotting Component Losses...")
    plot_component_losses(save_dir, epochs, train_stress, train_vel, test_stress, test_vel, model_cfg, train_cfg)

    print("Plotting Gradients...")
    plot_grad_norms(save_dir, epochs, grad_norms, model_cfg, train_cfg)

    # print("Plotting Weights...")
    # plot_weights(save_dir, model, model_cfg, train_cfg)
    #
    # print("Plotting Activations...")
    # plot_activations(save_dir, activations, model_cfg, train_cfg)

    print("Plotting Predictions vs True (Normalized)...")
    plot_pred_vs_true(save_dir, feature_labels, iter_targets_norm, iter_predictions_norm, model_cfg, train_cfg,
                      folder_name="pred_vs_true", suffix="(Normalized)")

    print("Plotting Residuals (Denormalized)...")
    plot_residuals(save_dir, feature_labels, iter_targets, iter_predictions, model_cfg, train_cfg,
                   folder_name="residuals", suffix="(Denorm)")

    # --- Plot Normalized ---
    print("Plotting Residuals (Normalized)...")
    plot_residuals(save_dir, feature_labels, iter_targets_norm, iter_predictions_norm, model_cfg, train_cfg,
                   folder_name="residuals_norm", suffix="(Normalized)")