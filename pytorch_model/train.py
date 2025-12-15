import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from data.defplate_dataset import DefPlateDataset, collate_unet
from model.model_entire import GraphUNet_DefPlate
from torch.optim.lr_scheduler import ExponentialLR
import time
from typing import List, Tuple
from dataclasses import dataclass
from helpers.evaluation_helper import run_final_evaluation
from helpers.helpers import (format_training_time, create_model_hyperparams, load_config, load_trajectories,
                             print_training_config, setup_paths, get_feature_indices, get_device)

# Constants
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1
DIM_OUT_VEL = 3
DIM_OUT_STRESS = 1

@dataclass
class TrainingHistory:
    """Container for tracking training metrics."""
    train_losses: List[float]
    val_losses: List[float]
    train_vel_losses: List[float]
    train_stress_losses: List[float]
    test_vel_losses: List[float]
    test_stress_losses: List[float]
    grad_norms: List[float]

    @classmethod
    def create_empty(cls) -> 'TrainingHistory':
        return cls([], [], [], [], [], [], [])

def _create_standard_dataloaders(dataset: DefPlateDataset, batch_size: int, shuffle: bool, num_workers: int,
                                pin_memory: bool) -> Tuple[DataLoader, DataLoader]:
    """Create train/test dataloaders with 80/20 split."""
    total = len(dataset)
    perm = torch.randperm(total)
    split = int(0.8 * total)

    train_idx = perm[:split]
    test_idx = perm[split:]

    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_unet,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_unet,
                             num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, test_loader

def _create_overfit_dataloader(dataset: DefPlateDataset, overfit_traj_id: int, overfit_time_idx_list: List[int]) -> DataLoader:
    """Create a dataloader for overfitting on specific samples."""
    overfit_indices = []

    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        if overfit_traj_id is not None and sample['traj_id'] != overfit_traj_id:
            continue
        if sample['time_idx'] in overfit_time_idx_list:
            overfit_indices.append(idx)

    if len(overfit_indices) == 0:
        raise ValueError(
            f"No samples found matching overfit criteria: "
            f"traj_id={overfit_traj_id}, time_idx={overfit_time_idx_list}"
        )

    overfit_set = Subset(dataset, overfit_indices)
    loader = DataLoader(overfit_set, batch_size=len(overfit_indices), shuffle=False, collate_fn=collate_unet)

    print(f"\nOverfitting on trajectory {overfit_traj_id} with {len(overfit_indices)} time steps")
    print_overfit_samples(loader)

    return loader


def compute_loss(adj_A_list: list, feat_tp1_mat_list: list, node_types_list: list, preds_list: list,
                 velocity_idxs: slice, stress_idxs: slice) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss per batch."""
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    num_graphs = len(adj_A_list)

    for pred, target, nodetype in zip(preds_list, feat_tp1_mat_list, node_types_list):
        vel_loss, stress_loss = _compute_single_graph_loss(pred, target, nodetype, velocity_idxs, stress_idxs)
        total_vel_loss += vel_loss
        total_stress_loss += stress_loss
        total_loss += vel_loss + stress_loss

    return (total_loss / num_graphs, total_vel_loss / num_graphs, total_stress_loss / num_graphs)


def _compute_single_graph_loss(pred: torch.Tensor, target: torch.Tensor, nodetype: torch.Tensor, velocity_idxs: slice,
    stress_idxs: slice) -> Tuple[float, float]:
    """Compute loss for a single graph."""
    vel_mask = (nodetype == NORMAL_NODE)
    stress_mask = (nodetype == NORMAL_NODE) | (nodetype == BOUNDARY_NODE)

    target_vel = target[:, velocity_idxs]
    target_stress = target[:, stress_idxs]
    pred_vel = pred[:, :3]
    pred_stress = pred[:, 3:4]

    vel_loss = 0.0
    stress_loss = 0.0

    if vel_mask.any():
        vel_loss = F.huber_loss(pred_vel[vel_mask], target_vel[vel_mask])

    if stress_mask.any():
        stress_loss = F.huber_loss(pred_stress[stress_mask], target_stress[stress_mask])

    return vel_loss, stress_loss


def _get_grad_norm(model: torch.nn.Module) -> float:
    """Get gradient norm of current batch (L2)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


@torch.no_grad()
def _validate_one_epoch(model: torch.nn.Module, test_loader: DataLoader, device: torch.device, velocity_idxs: slice,
                       stress_idxs: slice) -> Tuple[float, float, float]:
    """Run one validation epoch. Returns (avg_loss, avg_vel_loss, avg_stress_loss)."""
    model.eval()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0

    for batch in test_loader:
        adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, _, _, _, node_types, *_ = batch

        adj_mat_list = [A.to(device) for A in adj_mat_list]
        feat_t_mat_list = [X.to(device) for X in feat_t_mat_list]
        feat_tp1_mat_list = [X.to(device) for X in feat_tp1_mat_list]
        node_types = [nt.to(device) for nt in node_types]

        preds_list = model(adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, node_types)
        batch_loss, vel_loss, stress_loss = compute_loss(adj_mat_list, feat_tp1_mat_list, node_types, preds_list,
            velocity_idxs, stress_idxs)

        total_loss += batch_loss.item()
        total_vel_loss += vel_loss.item()
        total_stress_loss += stress_loss.item()

    n = len(test_loader)
    return total_loss / n, total_vel_loss / n, total_stress_loss / n


def _train_one_epoch(model: torch.nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer,
                    device: torch.device, velocity_idxs: slice, stress_idxs: slice) -> Tuple[float, float, float, float]:
    """Run one training epoch. Returns (avg_loss, avg_vel_loss, avg_stress_loss, avg_grad_norm)."""
    model.train()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for batch in train_loader:
        adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, _, _, _, node_types, *_ = batch

        # Move to device
        adj_mat_list = [A.to(device) for A in adj_mat_list]
        feat_t_mat_list = [X.to(device) for X in feat_t_mat_list]
        feat_tp1_mat_list = [X.to(device) for X in feat_tp1_mat_list]
        node_types = [nt.to(device) for nt in node_types]

        optimizer.zero_grad()

        preds_list = model(adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, node_types)
        batch_loss, vel_loss, stress_loss = compute_loss(
            adj_mat_list, feat_tp1_mat_list, node_types, preds_list,
            velocity_idxs, stress_idxs
        )

        batch_loss.backward()
        total_grad_norm += _get_grad_norm(model)
        optimizer.step()

        total_loss += batch_loss.item()
        total_vel_loss += vel_loss.item()
        total_stress_loss += stress_loss.item()
        num_batches += 1

    n = max(num_batches, 1)
    return total_loss / n, total_vel_loss / n, total_stress_loss / n, total_grad_norm / n

def train_gunet(device, num_workers, pin_memory):
    """Training loop"""
    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    # Extract model and training parameters
    model_cfg = config['model']
    train_cfg = config['training']
    # Load train config
    checkpoint_path, plots_dir = setup_paths(train_cfg)
    include_mesh_pos = "False" not in train_cfg['datapath']
    feat_idx = get_feature_indices(include_mesh_pos)
    torch.manual_seed(train_cfg['random_seed'])
    np.random.seed(train_cfg['random_seed'])

    print("\n=================================================")
    print(" LOADING PREPROCESSED DATA")
    print("=================================================\n")
    print(f"\t Preprocessed data: {train_cfg['datapath']}")

    # Load preprocessed trajectories
    if not os.path.exists(train_cfg['datapath']):
        raise FileNotFoundError(
            f"Preprocessed data not found at {train_cfg['datapath']}\n"
            f"Please run 'python preprocess_data.py' first to generate the preprocessed data."
        )

    list_of_trajs = load_trajectories(train_cfg['datapath'], train_cfg['num_train_trajs'])

    # Build dataset from these trajectories
    dataset = DefPlateDataset(list_of_trajs, add_world_edges=train_cfg['add_world_edges'],
                              world_pos_idxs=feat_idx.world_pos, velocity_idxs=feat_idx.velocity)
    print(f"Total training pairs (X_t, X_t+1): {len(dataset)}")

    # Create dataloaders based on mode
    if train_cfg['mode'] == "overfit":
        loader = _create_overfit_dataloader(dataset, train_cfg.get('overfit_traj_id'),
                                            train_cfg.get('overfit_time_idx', []))
        train_loader, test_loader = loader, loader
    else:
        train_loader, test_loader = _create_standard_dataloaders(dataset, train_cfg['batch_size'], train_cfg['shuffle'],
                                                                 num_workers, pin_memory)

    # Build model and optimizer
    model_hyperparams = create_model_hyperparams(model_cfg)
    model = GraphUNet_DefPlate(feat_idx.dim_in, DIM_OUT_VEL, DIM_OUT_STRESS, model_hyperparams).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['adam_weight_decay'])
    scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_lr_scheduler'])

    # Training
    print_training_config(train_cfg, train_loader)
    history = TrainingHistory.create_empty()
    start_time = time.time()

    for epoch in range(train_cfg['epochs']):
        # Train
        train_loss, train_vel, train_stress, grad_norm = _train_one_epoch(model, train_loader, optimizer, device,
                                                                          feat_idx.velocity, feat_idx.stress)

        # Validate
        val_loss, val_vel, val_stress = _validate_one_epoch(model, test_loader, device, feat_idx.velocity,
                                                            feat_idx.stress)

        # Record history
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_vel_losses.append(train_vel)
        history.train_stress_losses.append(train_stress)
        history.test_vel_losses.append(val_vel)
        history.test_stress_losses.append(val_stress)
        history.grad_norms.append(grad_norm)

        scheduler.step()

        print(f"[Train] [Epoch {epoch:03d}] "
            f"Train Loss: {train_loss:.6f} | Test Loss: {val_loss:.6f} | "
            f"Vel Loss: {train_vel:.6f} | Stress Loss: {train_stress:.6f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}")

    # Finish up
    total_time = time.time() - start_time
    print(f"\n[train] Total training time: {format_training_time(total_time)}")

    # Save model
    print(f"\n[train] Saving model to {checkpoint_path}")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    return model, test_loader, history, feat_idx, plots_dir

if __name__ == "__main__":
    num_workers = 8
    pin_memory = True
    device = get_device()
    model, test_loader, history, feat_idx, plots_dir = train_gunet(device, num_workers, pin_memory)
    run_final_evaluation(model, test_loader, device, history, feat_idx.velocity, feat_idx.stress, plots_dir)
