import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import textwrap
import yaml

def load_config(config_path):
    """Load model and training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def make_final_plots(save_dir, train_losses, val_losses, metric_name, train_metrics, val_metrics, grad_norms, model,
                     activations, predictions, targets, train_vel_losses, train_stress_losses, test_vel_losses,
                     test_stress_losses):
    """
    Generates and saves the requested plots.

    :param save_dir: str
        Directory to save plots.
    :param train_losses: List
        List of training losses per epoch.
    :param val_losses: List
        List of validation losses per epoch.
    :param metric_name: str
        Name of the metric (e.g., 'MAE').
    :param train_metrics: List
        List of training metrics per epoch.
    :param val_metrics: List
        List of validation metrics per epoch.
    :param grad_norms: List
        List of average gradient norms per epoch.
    :param model: torch.nn.Module
        The trained model.
    :param activations: Dict
        Dictionary of {layer_name: numpy_array} for activation distributions.
    :param predictions: numpy.ndarray
        Predictions array of shape [N, D].
    :param targets: numpy.ndarray
        Targets array of shape [N, D].
    """

    # --------- Helper to make sure everything is on CPU + NumPy ----------
    def to_cpu_numpy(x):
        """
        Convert Torch tensors (any device), lists/tuples of tensors, or plain lists
        to a NumPy array on CPU.
        """
        # Single tensor
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()

        # List / tuple
        if isinstance(x, (list, tuple)):
            if len(x) == 0:
                return np.array([])

            # If elements are tensors, move each to CPU
            if isinstance(x[0], torch.Tensor):
                out = []
                for t in x:
                    t = t.detach().cpu()
                    # scalar tensor -> float
                    if t.dim() == 0:
                        out.append(t.item())
                    else:
                        out.append(t.numpy())
                return np.array(out, dtype=object if any(isinstance(e, np.ndarray) for e in out) else float)

            # Otherwise assume it's already numeric-ish
            return np.asarray(x)

        # Fallback
        return np.asarray(x)

    # Convert 1D training history stuff
    train_losses = to_cpu_numpy(train_losses)
    val_losses = to_cpu_numpy(val_losses)

    if train_metrics is not None:
        train_metrics = to_cpu_numpy(train_metrics)
    if val_metrics is not None:
        val_metrics = to_cpu_numpy(val_metrics)

    grad_norms = to_cpu_numpy(grad_norms)
    train_stress_losses = to_cpu_numpy(train_stress_losses)
    train_vel_losses = to_cpu_numpy(train_vel_losses)
    test_stress_losses = to_cpu_numpy(test_stress_losses)
    test_vel_losses = to_cpu_numpy(test_vel_losses)

    # Predictions / targets may also be tensors
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.detach().cpu().numpy()
    if isinstance(targets, torch.Tensor):
        targets = targets.detach().cpu().numpy()

    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    # Extract model and training parameters
    model_cfg = config['model']
    train_cfg = config['training']


    # ------------------------------------------------------------------
    # Helper: flatten config dicts and add them as small footer text
    # ------------------------------------------------------------------
    def _flatten_dict(d, parent_key=""):
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else str(k)
            if isinstance(v, dict):
                items.extend(_flatten_dict(v, new_key).items())
            else:
                items.append((new_key, v))
        return dict(items)

    flat_model = _flatten_dict(model_cfg, "model")
    flat_train = _flatten_dict(train_cfg, "train")

    footer_items = list(flat_model.items()) + list(flat_train.items())
    footer_text = ", ".join(f"{k}={v}" for k, v in footer_items)

    def add_cfg_footer():
        """Add a small text block with model_cfg and train_cfg at the bottom."""
        if not footer_text:
            return
        # Wrap to multiple lines so it doesn’t go off the figure
        wrapped = textwrap.fill(footer_text, width=150)
        # Make space at the bottom for the text
        plt.subplots_adjust(bottom=0.25)
        # Put text in small font at the very bottom
        plt.figtext(0.5, 0.01, wrapped,
                    ha='center', va='bottom', fontsize=6)

    # Training and validation loss and metric vs epoch
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Train Loss')
    plt.plot(epochs, val_losses, label='Val Loss')
    if train_metrics is not None and val_metrics is not None:
        plt.plot(epochs, train_metrics, label=f'Train {metric_name}', linestyle='--')
        plt.plot(epochs, val_metrics, label=f'Val {metric_name}', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.ylim(top=0.5, bottom=0)
    plt.title('Training/Validation Loss and Metric vs Epoch')
    plt.legend()
    plt.grid(True)
    add_cfg_footer()
    plt.savefig(os.path.join(save_dir, 'loss_metric_vs_epoch.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_stress_losses, label='Train stress loss')
    plt.plot(epochs, train_vel_losses, label='Train velocity loss')
    plt.plot(epochs, test_stress_losses, label='Test stress loss')
    plt.plot(epochs, test_vel_losses, label='Test velocity loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.ylim(top=1, bottom=0)
    plt.title('Stress and velocity losses, test and train')
    plt.legend()
    plt.grid(True)
    add_cfg_footer()
    plt.savefig(os.path.join(save_dir, 'stress_vel_losses.png'))
    plt.close()

    # Gradient norm averaged per epoch
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, grad_norms, label='Gradient Norm')
    plt.xlabel('Epoch')
    plt.ylabel('Gradient Norm')
    plt.ylim(top=1, bottom=0)
    plt.title('Average Gradient Norm per Epoch')
    plt.legend()
    plt.grid(True)
    add_cfg_footer()
    plt.savefig(os.path.join(save_dir, 'grad_norm_vs_epoch.png'))
    plt.close()

    # Final weight histograms
    weights_dir = os.path.join(save_dir, 'weights')
    os.makedirs(weights_dir, exist_ok=True)

    for name, param in model.named_parameters():
        if param.requires_grad:
            plt.figure(figsize=(8, 5))
            data = param.data.cpu().numpy().flatten()
            plt.hist(data, bins=50, alpha=0.7)
            plt.title(f'Weight Distribution - {name}')
            plt.grid(True)
            add_cfg_footer()
            clean_name = name.replace('.', '_')
            plt.savefig(os.path.join(weights_dir, f'{clean_name}.png'))
            plt.close()

    # Activation distributions
    acts_dir = os.path.join(save_dir, 'activations')
    os.makedirs(acts_dir, exist_ok=True)

    for name, acts in activations.items():
        plt.figure(figsize=(8, 5))
        plt.hist(acts.flatten(), bins=50, alpha=0.7)
        plt.title(f'Activation Distribution - {name}')
        plt.grid(True)
        add_cfg_footer()
        clean_name = name.replace('.', '_')
        plt.savefig(os.path.join(acts_dir, f'{clean_name}.png'))
        plt.close()

    # Predicted vs. true scatter plot
    if isinstance(predictions, list):
        num_features = len(predictions)
        feature_labels = [f'Feature {i}' for i in range(num_features)]
        if num_features == 4:
            feature_labels = ['Vel X', 'Vel Y', 'Vel Z', 'Stress']

        iter_predictions = predictions
        iter_targets = targets
    else:
        num_features = predictions.shape[1]
        feature_labels = [f'Feature {i}' for i in range(num_features)]
        if num_features == 4:
            feature_labels = ['Vel X', 'Vel Y', 'Vel Z', 'Stress']

        iter_predictions = [predictions[:, i] for i in range(num_features)]
        iter_targets = [targets[:, i] for i in range(num_features)]

    pred_dir = os.path.join(save_dir, 'pred_vs_true')
    os.makedirs(pred_dir, exist_ok=True)

    for i in range(num_features):
        plt.figure(figsize=(8, 8))
        y_true = iter_targets[i]
        y_pred = iter_predictions[i]

        # Scatter with subsampling if too many points
        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            plt.scatter(y_true[idx], y_pred[idx], alpha=0.3, s=5)
        else:
            plt.scatter(y_true, y_pred, alpha=0.5, s=10)

        # Diagonal line
        if len(y_true) > 0:
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8)

        plt.xlabel('True Value')
        plt.ylabel('Predicted Value')
        plt.title(f'Predicted vs True - {feature_labels[i]}')
        plt.grid(True)
        add_cfg_footer()
        plt.savefig(os.path.join(pred_dir, f'feat_{i}.png'))
        plt.close()

    # Residuals vs. true value
    res_dir = os.path.join(save_dir, 'residuals')
    os.makedirs(res_dir, exist_ok=True)

    for i in range(num_features):
        plt.figure(figsize=(10, 6))
        y_true = iter_targets[i]
        y_pred = iter_predictions[i]
        res = y_pred - y_true

        if len(y_true) > 10000:
            idx = np.random.choice(len(y_true), 10000, replace=False)
            plt.scatter(y_true[idx], res[idx], alpha=0.3, s=5)
        else:
            plt.scatter(y_true, res, alpha=0.5, s=10)

        plt.axhline(0, color='r', linestyle='--')
        plt.xlabel('True Value')
        plt.ylabel('Residual (Pred - True)')
        plt.title(f'Residuals vs True - {feature_labels[i]}')
        plt.grid(True)
        add_cfg_footer()
        plt.savefig(os.path.join(res_dir, f'feat_{i}.png'))
        plt.close()
