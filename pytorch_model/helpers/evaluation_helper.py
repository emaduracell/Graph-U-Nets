import torch
import numpy as np
from typing import Tuple
from helpers.plots import make_final_plots

BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1
DIM_OUT_VEL = 3
DIM_OUT_STRESS = 1


def _collect_evaluation_data(preds_list, targets, node_types_gpu, means, stds, velocity_idxs, stress_idxs, device,
                             denorm_data, norm_data):
    """
    Collect denormalized and normalized prediction data.

    Args:
        preds_list: List of tensors containing the model predictions for the current batch.
        targets: List of tensors containing the ground truth target values.
        node_types_gpu: List of tensors indicating the node types, moved to the device.
        means: List of mean values used for data normalization.
        stds: List of standard deviation values used for data normalization.
        velocity_idxs: Slice or indices identifying velocity components in the data.
        stress_idxs: Slice or indices identifying stress components in the data.
        device: The computation device (CPU or GPU) to be used.
        denorm_data: Dictionary to accumulate the denormalized velocity and stress data.
        norm_data: Dictionary to accumulate the normalized velocity and stress data.
    """
    for pred, target, nodetype, mean, std in zip(preds_list, targets, node_types_gpu, means, stds):
        mean = mean.to(device).squeeze()
        std = std.to(device).squeeze()

        # Denormalization stats
        std_pred = torch.cat([std[velocity_idxs], std[stress_idxs]])
        mean_pred = torch.cat([mean[velocity_idxs], mean[stress_idxs]])

        pred_denorm = pred * std_pred + mean_pred
        target_denorm = target * std + mean

        eval_mask = (nodetype == NORMAL_NODE)
        if not eval_mask.any():
            continue

        # Collect denormalized
        denorm_data['vel_preds'].append(pred_denorm[:, :3][eval_mask].cpu().numpy())
        denorm_data['vel_targets'].append(target_denorm[:, velocity_idxs][eval_mask].cpu().numpy())
        denorm_data['stress_preds'].append(pred_denorm[:, 3:4][eval_mask].cpu().numpy())
        denorm_data['stress_targets'].append(target_denorm[:, stress_idxs][eval_mask].cpu().numpy())

        # Collect normalized
        norm_data['vel_preds'].append(pred[:, :3][eval_mask].cpu().numpy())
        norm_data['vel_targets'].append(target[:, velocity_idxs][eval_mask].cpu().numpy())
        norm_data['stress_preds'].append(pred[:, 3:4][eval_mask].cpu().numpy())
        norm_data['stress_targets'].append(target[:, stress_idxs][eval_mask].cpu().numpy())


def _prepare_plot_data(data):
    """Prepare concatenated data lists for plotting."""

    def concat_or_empty(preds_list, targets_list, dim):
        if preds_list:
            return np.concatenate(preds_list, axis=0), np.concatenate(targets_list, axis=0)
        return np.zeros((0, dim)), np.zeros((0, dim))

    vel_preds, vel_targets = concat_or_empty(data['vel_preds'], data['vel_targets'], 3)
    stress_preds, stress_targets = concat_or_empty(data['stress_preds'], data['stress_targets'], 1)

    preds = [vel_preds[:, 0], vel_preds[:, 1], vel_preds[:, 2], stress_preds[:, 0]]
    targets = [vel_targets[:, 0], vel_targets[:, 1], vel_targets[:, 2], stress_targets[:, 0]]

    return preds, targets


@torch.no_grad()
def run_final_evaluation(model, test_loader, device, history, velocity_idxs, stress_idxs, plots_dir, config_path):
    """
    Run evaluation and generate final plots.

    Args:
        model: torch.nn.Module, the trained model to be evaluated.
        test_loader: DataLoader containing the test dataset batches.
        device: torch.device, the computation device to use.
        history: Object containing training metrics (losses, gradients) for visualization.
        velocity_idxs: slice, indices to extract velocity features.
        stress_idxs: slice, indices to extract stress features.
        plots_dir: str, directory path where plots will be saved.
        config_path: str, path to the configuration file used.
    """
    print("[train] Generating final evaluation plots...")

    activations = {}
    # Support both GraphUNet (velocity_mlp) and EGNN (phi_v) models
    if hasattr(model, 'velocity_mlp'):
        velocity_head = model.velocity_mlp
    elif hasattr(model, 'phi_v'):
        velocity_head = model.phi_v
    else:
        raise AttributeError(f"Model {type(model)} does not have 'velocity_mlp' or 'phi_v' attribute")
    
    handle = velocity_head.register_forward_hook(
        lambda m, i, o: activations.update({'latent_features': i[0].detach().cpu().numpy()})
    )

    # Containers for predictions and targets
    denorm_data = {'vel_preds': [], 'vel_targets': [], 'stress_preds': [], 'stress_targets': []}
    norm_data = {'vel_preds': [], 'vel_targets': [], 'stress_preds': [], 'stress_targets': []}

    model.eval()
    for i, batch in enumerate(test_loader):
        print(f"[run_final_evaluation] batch {i}")
        adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, _, node_types, _, time_indices = batch

        # Adjacency is already sliced to [N, N] in the dataset/collate
        gs = [A.to(device) for A in adj_mat_list]
        hs = [X.to(device) for X in feat_t_mat_list]
        targets = [X.to(device) for X in feat_tp1_mat_list]
        node_types_gpu = [nt.to(device) for nt in node_types]

        preds_list = model(gs, hs)

        if i == 0:
            handle.remove()

        _collect_evaluation_data(preds_list, targets, node_types_gpu, means, stds,
                                 velocity_idxs, stress_idxs, device, denorm_data, norm_data)

    # Prepare data for plotting
    final_preds, final_targets = _prepare_plot_data(denorm_data)
    final_preds_norm, final_targets_norm = _prepare_plot_data(norm_data)

    make_final_plots(save_dir=plots_dir, train_losses=history.train_losses, val_losses=history.val_losses,
                     grad_norms=history.grad_norms, model=model, activations=activations,
                     predictions=final_preds, targets=final_targets, predictions_norm=final_preds_norm,
                     targets_norm=final_targets_norm, train_vel_losses=history.train_vel_losses,
                     train_stress_losses=history.train_stress_losses, test_vel_losses=history.test_vel_losses,
                     test_stress_losses=history.test_stress_losses, velocity_idxs=velocity_idxs,
                     stress_idxs=stress_idxs, config_path=config_path)
    print(f"Plots saved to {plots_dir}")
