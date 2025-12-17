import torch
import torch.optim as optim
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
import os
import time
from typing import List, Optional
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim.lr_scheduler import ExponentialLR
from torch.amp import autocast, GradScaler

# Helpers and Data
from helpers.helpers import (format_training_time, load_config, print_training_config,
                             setup_paths, get_feature_indices, get_device, print_overfit_samples)
from data_builder_pyg import TrajectoryDataset
from model.pyg_gunet_wrapper import GraphUNet_DefPlate

# Constants
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1
DIM_OUT_VEL = 3
DIM_OUT_STRESS = 1


@dataclass
class TrainingHistory:
    train_losses: List[float]
    val_losses: List[float]
    train_vel_losses: List[float]
    train_stress_losses: List[float]
    test_vel_losses: List[float]
    test_stress_losses: List[float]
    grad_norms: List[float]

    @classmethod
    def create_empty(cls):
        return cls([], [], [], [], [], [], [])


def _create_dataloaders(dataset, batch_size, shuffle, num_workers, mode, overfit_traj_id=None):
    """
    Create dataloaders. Handles both Standard 80/20 split and Overfit mode.
    """
    total = len(dataset)

    if mode == "overfit":
        # Select specific trajectory
        if overfit_traj_id is not None:
            if overfit_traj_id >= total:
                raise ValueError(f"overfit_traj_id {overfit_traj_id} out of bounds (total: {total})")
            indices = [overfit_traj_id]
        else:
            indices = [0]  # Default to first

        # Create subset
        subset = dataset[indices]

        # For overfit, we usually want batch_size = number of samples (full batch overfit)
        loader = DataLoader(subset, batch_size=len(indices), shuffle=False,
                            num_workers=num_workers,
                            pin_memory=False)  # Pin memory False if overfit (usually on device)

        print(f"\n[Data] Overfitting on Trajectory ID: {indices[0]}")
        return loader, loader  # Train on set, Test on set (to see convergence)

    else:
        # Standard Split
        perm = torch.randperm(total)
        split = int(0.8 * total)

        train_dataset = dataset[perm[:split]]
        test_dataset = dataset[perm[split:]]

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle,
                                  num_workers=num_workers, pin_memory=(num_workers > 0))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=(num_workers > 0))

        return train_loader, test_loader


def _compute_batch_loss(pred_seq, target_seq, node_type, velocity_idxs, stress_idxs):
    """
    Compute masked Huber loss for Velocity and Stress.
    """
    # Flatten Time and Nodes: (Batch * T * N, F) or (T * N, F)
    pred_flat = pred_seq.view(-1, pred_seq.shape[-1])
    target_flat = target_seq.view(-1, target_seq.shape[-1])

    total_steps = pred_flat.shape[0]
    total_nodes_spatial = node_type.shape[0]

    # Time steps effectively present in the flattened prediction
    T_effective = total_steps // total_nodes_spatial

    # Repeat node types T times
    node_type_flat = node_type.repeat(T_effective)

    # Masks
    vel_mask = (node_type_flat == NORMAL_NODE)
    stress_mask = (node_type_flat == NORMAL_NODE) | (node_type_flat == BOUNDARY_NODE)

    # Slices
    target_vel = target_flat[:, velocity_idxs]
    target_stress = target_flat[:, stress_idxs]
    pred_vel = pred_flat[:, :3]
    pred_stress = pred_flat[:, 3:4]

    vel_loss = 0.0
    stress_loss = 0.0

    if vel_mask.any():
        vel_loss = F.huber_loss(pred_vel[vel_mask], target_vel[vel_mask])

    if stress_mask.any():
        stress_loss = F.huber_loss(pred_stress[stress_mask], target_stress[stress_mask])

    return vel_loss, stress_loss


def _get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def _slice_time_sequence(x_seq, pos_seq, time_indices: Optional[List[int]]):
    """
    Slices the (Batch, T, N, F) tensors based on time indices.
    Returns Input (t) and Target (t+1).
    """
    # Determine Time Dimension
    t_dim = 1 if x_seq.dim() == 4 else 0
    max_t = x_seq.shape[t_dim]

    if time_indices is not None:
        # Overfit mode: pick specific t for input, t+1 for target
        # Ensure indices + 1 are valid
        valid_indices = [t for t in time_indices if t + 1 < max_t]
        if not valid_indices:
            raise ValueError("No valid time indices for training (t+1 exceeds bounds).")

        idx_t = torch.tensor(valid_indices, device=x_seq.device)
        idx_tp1 = torch.tensor([t + 1 for t in valid_indices], device=x_seq.device)

        if t_dim == 1:
            x_input = x_seq.index_select(1, idx_t)
            x_target = x_seq.index_select(1, idx_tp1)
            pos_input = pos_seq.index_select(1, idx_t)
        else:
            x_input = x_seq.index_select(0, idx_t)
            x_target = x_seq.index_select(0, idx_tp1)
            pos_input = pos_seq.index_select(0, idx_t)

    else:
        # Standard mode: Input 0..T-1, Target 1..T
        if t_dim == 1:
            x_input = x_seq[:, :-1]
            x_target = x_seq[:, 1:]
            pos_input = pos_seq[:, :-1]
        else:
            x_input = x_seq[:-1]
            x_target = x_seq[1:]
            pos_input = pos_seq[:-1]

    return x_input, x_target, pos_input


@torch.no_grad()
def _validate_one_epoch(model, test_loader, device, velocity_idxs, stress_idxs, amp_enabled, time_indices=None):
    model.eval()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0

    for batch in tqdm(test_loader, desc="Val", leave=False):
        batch = batch.to(device)

        # Slice Time
        x_input_seq, x_target_seq, pos_input = _slice_time_sequence(batch.x, batch.pos_seq, time_indices)

        if device.type == 'cuda':
            with autocast(device_type=device.type, enabled=amp_enabled):
                preds = model(x_input_seq, batch.edge_index, batch.batch)
                vel_loss, stress_loss = _compute_batch_loss(preds, x_target_seq, batch.node_type,
                                                            velocity_idxs, stress_idxs)
        else:
            preds = model(x_input_seq, batch.edge_index, batch.batch)
            vel_loss, stress_loss = _compute_batch_loss(preds, x_target_seq, batch.node_type,
                                                        velocity_idxs, stress_idxs)

        total_loss += (vel_loss + stress_loss).item()
        total_vel_loss += vel_loss.item()
        total_stress_loss += stress_loss.item()

    n = len(test_loader)
    # Avoid div by zero if loader empty (unlikely)
    n = max(n, 1)
    return total_loss / n, total_vel_loss / n, total_stress_loss / n


def _train_one_epoch(model, train_loader, optimizer, device, velocity_idxs, stress_idxs, amp_enabled, scaler,
                     time_indices=None):
    model.train()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    total_grad_norm = 0.0

    for batch in tqdm(train_loader, desc="Train", leave=False):
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)

        # Slice Time (Handling Overfit vs Standard)
        x_input_seq, x_target_seq, pos_input = _slice_time_sequence(batch.x, batch.pos_seq, time_indices)

        if device.type == 'cuda':
            with autocast(device_type=device.type, enabled=amp_enabled):
                preds = model(x_input_seq, batch.edge_index, batch.batch)
                vel_loss, stress_loss = _compute_batch_loss(preds, x_target_seq, batch.node_type, velocity_idxs,
                                                            stress_idxs)
                loss = vel_loss + stress_loss
        else:
            preds = model(x_input_seq, batch.edge_index, batch.batch)
            vel_loss, stress_loss = _compute_batch_loss(preds, x_target_seq, batch.node_type, velocity_idxs,
                                                        stress_idxs)
            loss = vel_loss + stress_loss

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        if device.type != "cuda":
            total_grad_norm += _get_grad_norm(model)

        total_loss += loss.detach()
        total_vel_loss += vel_loss.detach()
        total_stress_loss += stress_loss.detach()

    n = len(train_loader)
    n = max(n, 1)
    avg_grad_norm = total_grad_norm / n if device.type != "cuda" else 0.0
    return total_loss.item() / n, total_vel_loss.item() / n, total_stress_loss.item() / n, avg_grad_norm


def train_pyg(device, num_workers):
    # Load Main Config
    config_path = os.path.join(os.path.dirname(__file__), "pyg_config.yaml")
    config = load_config(config_path)
    model_cfg = config['model']
    train_cfg = config['training']

    checkpoint_path, plots_dir = setup_paths(train_cfg)

    # 1. Load the Data Config to find the 'dataset_root'
    # Assuming datapath in train config points to the folder containing pyg_dataconfig.yaml
    dataconfig_path = "pyg_dataconfig.yaml"  # Or path relative to train script
    if not os.path.exists(dataconfig_path):
        raise FileNotFoundError(f"Could not find data config at {dataconfig_path}")
    dataconfig = load_config(dataconfig_path)
    # 2. Get Root
    dataset_root = dataconfig['output_dir']

    include_mesh_pos = dataconfig['include_mesh_pos']
    feat_idx = get_feature_indices(include_mesh_pos)

    # Flags
    amp_enabled = bool(train_cfg['amp'])
    move_all_to_device = bool(train_cfg["move_all_to_device"])
    mode = train_cfg['mode']
    overfit_time_idx = train_cfg['overfit_time_idx'] if mode == 'overfit' else None

    print("\n=================================================")
    print(" LOADING PYG DATASET")
    print(f" Source: {dataset_root}")
    print("=================================================\n")

    # 3. Instantiate Dataset
    if not os.path.exists(os.path.join(dataset_root, 'processed', 'pyg_processed_data.pt')):
        raise RuntimeError(f"Processed data not found in {dataset_root}. Please run data_builder_pyg.py first.")

    dataset = TrajectoryDataset(root=dataset_root, dataconfig=dataconfig, pre_transform=None, transform=None)
    print(f"Dataset loaded successfully. Num trajectories: {len(dataset)}")

    if move_all_to_device:
        print(f"[Data] Moving entire dataset to {device} for speed...")
        dataset.data = dataset.data.to(device)
        num_workers = 0

    # Create Dataloaders
    train_loader, test_loader = _create_dataloaders(
        dataset,
        train_cfg['batch_size'],
        train_cfg['shuffle'],
        num_workers,
        mode=mode,
        overfit_traj_id=train_cfg.get('overfit_traj_id')
    )

    if mode == 'overfit':
        print(f"[Overfit] Using time indices: {overfit_time_idx}")

    # Prepare Hyperparams for Model
    hyperparams = {
        'activation_mlps_final': model_cfg['activation_mlps_final'],
        'dropout_mlps_final': model_cfg['dropout_mlps_final'],
        'hid_mlp_dim': model_cfg['hid_mlp_dim']
    }

    # Initialize Model
    model = GraphUNet_DefPlate(in_channels=feat_idx.dim_in, hidden_channels=model_cfg['hidden_dim'],
                               depth=model_cfg.get('depth'), pool_ratios=model_cfg.get('pool_ratios'),
                               model_config_hyperparams=hyperparams).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['adam_weight_decay'],
                           fused=(device.type == "cuda"))
    scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_lr_scheduler'])
    scaler = GradScaler(enabled=amp_enabled)

    print_training_config(train_cfg, train_loader)
    history = TrainingHistory.create_empty()
    start_time = time.time()

    for epoch in range(train_cfg['epochs']):
        t_loss, t_vel, t_stress, grad = _train_one_epoch(model, train_loader, optimizer, device, feat_idx.velocity,
                                                         feat_idx.stress, amp_enabled, scaler,
                                                         time_indices=overfit_time_idx)

        v_loss, v_vel, v_stress = _validate_one_epoch(model, test_loader, device, feat_idx.velocity, feat_idx.stress,
                                                      amp_enabled, time_indices=overfit_time_idx)

        history.train_losses.append(t_loss)
        history.val_losses.append(v_loss)
        scheduler.step()

        tqdm.write(f"[Epoch {epoch:03d}] Train: {t_loss:.6f} | Val: {v_loss:.6f} | Vel: {t_vel:.6f} |"
                   f" Stress: {t_stress:.6f}")

    total_time = time.time() - start_time
    print(f"\n[train] Total training time: {format_training_time(total_time)}")

    print(f"\n[train] Saving model to {checkpoint_path}")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)


if __name__ == "__main__":
    device = get_device(cuda=False)
    # If using move_all_to_device, num_workers will be forced to 0 automatically
    train_pyg(device, num_workers=4)