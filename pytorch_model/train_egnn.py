import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from data_helper.defplate_dataset import DefPlateDataset, collate_unet
from model_egnn.egnn_deforming_plate import EGNN_DefPlate
from torch.optim.lr_scheduler import ExponentialLR
import time
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from helpers.evaluation_helper import run_final_evaluation
from helpers.helpers import (format_training_time, create_model_hyperparams, load_config, load_trajectories_preprocessed,
                             print_training_config, setup_paths, get_feature_indices, get_device, print_overfit_samples,
                             move_any_to_device)
from torch.amp import autocast, GradScaler
import argparse

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

def _create_standard_dataloaders(dataset, batch_size, shuffle, num_workers, pin_memory):
    """
    Create train/test dataloaders with 80/20 split.

    Args:
        dataset: DefPlateDataset
        batch_size: int
        shuffle: bool
        num_workers: int
        pin_memory: bool

    :return: Tuple[DataLoader, DataLoader]
    """
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

def _create_overfit_dataloader(dataset, overfit_traj_id, overfit_time_idx_list):
    """
    Create a dataloader for overfitting on specific samples.

    Args:
        dataset: DefPlateDataset
        overfit_traj_id: int
        overfit_time_idx_list: List[int]

    :return: DataLoader
    """

    overfit_indices = []

    for idx in range(len(dataset)):
        sample = dataset.samples[idx]
        if overfit_traj_id is not None and sample['traj_id'] != overfit_traj_id:
            continue
        if sample['time_idx'] in overfit_time_idx_list:
            overfit_indices.append(idx)

    if len(overfit_indices) == 0:
        raise ValueError(f"No samples found matching overfit criteria: "
                         f"traj_id={overfit_traj_id}, time_idx={overfit_time_idx_list}")

    overfit_set = Subset(dataset, overfit_indices)
    loader = DataLoader(overfit_set, batch_size=len(overfit_indices), shuffle=False, collate_fn=collate_unet)

    print(f"\nOverfitting on trajectory {overfit_traj_id} with {len(overfit_indices)} time steps")
    print_overfit_samples(loader)

    return loader


def compute_loss(adj_A_list, feat_tp1_mat_list, node_types_list, preds_list, velocity_idxs, stress_idxs):
    """
    Compute loss per batch.

    Args:
        adj_A_list: list
        feat_tp1_mat_list: list
        node_types_list: list
        preds_list: list
        velocity_idxs: slice
        stress_idxs: slice
    :return: (total_loss / num_graphs, total_vel_loss / num_graphs, total_stress_loss / num_graphs)
        every element of the tuple is a torch.Tensor
    """
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


def _compute_single_graph_loss(pred, target, nodetype, velocity_idxs,
    stress_idxs):
    """
    Compute loss for a single graph.

    Args:
        pred: torch.Tensor
        target: torch.Tensor
        nodetype: torch.Tensor
        velocity_idxs: slice
        stress_idxs: slice

    :return: (vel_loss, stress_loss)
    """
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


def _get_grad_norm(model):
    """Get gradient norm of current batch (L2)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


@torch.no_grad()
def _validate_one_epoch(model, test_loader, device, velocity_idxs, stress_idxs, amp_enabled: bool):
    """
    Run one validation epoch.

    Args:
        model: torch.nn.Module
        test_loader: DataLoader
        device: torch.device
        velocity_idxs: slice
        stress_idxs: slice

    :return: (avg_loss, avg_vel_loss, avg_stress_loss)
    """
    model.eval()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0

    for batch in tqdm(test_loader, desc="Val", leave=False):
        adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, _, _, _, node_types, _, time_indices = batch

        # Adjacency is already sliced to [N, N] in the dataset/collate
        adj_mat_list = [A.to(device, non_blocking=True) for A in adj_mat_list]
        feat_t_mat_list = [X.to(device, non_blocking=True) for X in feat_t_mat_list]
        feat_tp1_mat_list = [X.to(device, non_blocking=True) for X in feat_tp1_mat_list]
        node_types = [nt.to(device, non_blocking=True) for nt in node_types]

        if device.type == 'cuda':
            with autocast(device_type=device.type, enabled=amp_enabled):
                preds_list = model(adj_mat_list, feat_t_mat_list)
        else:
            preds_list = model(adj_mat_list, feat_t_mat_list)
        batch_loss, vel_loss, stress_loss = compute_loss(adj_mat_list, feat_tp1_mat_list, node_types, preds_list,
            velocity_idxs, stress_idxs)

        # Call item only at the end for computational reasons
        total_loss += batch_loss.detach()
        total_vel_loss += vel_loss.detach()
        total_stress_loss += stress_loss.detach()

    n = len(test_loader)
    return total_loss.item() / n, total_vel_loss.item() / n, total_stress_loss.item() / n


def _train_one_epoch(model, train_loader, optimizer, device, velocity_idxs, stress_idxs, amp_enabled, scaler,
                     move_all_to_device):
    """
    Run one training epoch. Returns (avg_loss, avg_vel_loss, avg_stress_loss, avg_grad_norm).

    Args:
        model: torch.nn.Module
        train_loader: DataLoader
        optimizer: torch.optim.Optimizer
        device: torch.device
        velocity_idxs: slice
        stress_idxs: slice
        amp_enabled: bool
        scaler: GradScaler | None

    :return: (total_loss, total_vel_loss, total_stress_loss, total_grad_norm)
        floats of averaged loss for that epoch
    """
    model.train()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Train", leave=False):
        adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, _, _, _, node_types, _, time_indices = batch

        # Adjacency is already sliced to [N, N] in the dataset/collate
        if not move_all_to_device and adj_mat_list[0].device != device:
            # Move to device
            adj_mat_list = [A.to(device) for A in adj_mat_list]
            feat_t_mat_list = [X.to(device) for X in feat_t_mat_list]
            feat_tp1_mat_list = [X.to(device) for X in feat_tp1_mat_list]
            node_types = [nt.to(device) for nt in node_types]

        optimizer.zero_grad(set_to_none=True)

        if device.type == 'cuda':
            with autocast(device_type=device.type, enabled=amp_enabled):
                preds_list = model(adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, node_types)
                batch_loss, vel_loss, stress_loss = compute_loss(
                    adj_mat_list, feat_tp1_mat_list, node_types, preds_list,
                    velocity_idxs, stress_idxs
                )
        else:
            preds_list = model(adj_mat_list, feat_t_mat_list)
            batch_loss, vel_loss, stress_loss = compute_loss(
                adj_mat_list, feat_tp1_mat_list, node_types, preds_list,
                velocity_idxs, stress_idxs
            )

        if amp_enabled and scaler is not None:
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            optimizer.step()

        if device.type != "cuda":
            total_grad_norm += _get_grad_norm(model)

        # For speed reasons, do not call .item() already here
        total_loss += batch_loss.detach()
        total_vel_loss += vel_loss.detach()
        total_stress_loss += stress_loss.detach()
        num_batches += 1

    n = max(num_batches, 1)
    avg_grad_norm = total_grad_norm / n
    return total_loss.item() / n, total_vel_loss.item() / n, total_stress_loss.item() / n, avg_grad_norm

def train_egnn(device, num_workers, pin_memory, config_path=None):
    """Training loop"""

    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config_egnn.yaml")
    config = load_config(config_path)

    # Extract model and training parameters
    model_cfg = config['model']
    train_cfg = config['training']
    
    base_dir = os.path.dirname(os.path.dirname(__file__))
    datapath = os.path.join(base_dir, train_cfg['datapath'])
    
    checkpoint_path, plots_dir = setup_paths(train_cfg)
    dataconfig = load_config(os.path.join(datapath, 'used_dataconfig.yaml'))
    include_mesh_pos = dataconfig['include_mesh_pos']
    feat_idx = get_feature_indices(include_mesh_pos)
    torch.manual_seed(train_cfg['random_seed'])
    np.random.seed(train_cfg['random_seed'])
    move_all_to_device = bool(train_cfg.get("move_all_to_device"))

    print("\n=================================================")
    print(" LOADING PREPROCESSED DATA")
    print("=================================================\n")
    print(f"\t Preprocessed data: {datapath}")

    # Load preprocessed trajectories
    if not os.path.exists(datapath):
        raise FileNotFoundError(
            f"Preprocessed data not found at {datapath}\n"
            f"Please run 'python preprocess_data.py' first to generate the preprocessed data."
        )

    list_of_trajs = load_trajectories_preprocessed(
        os.path.join(datapath, "preprocessed_train.pt"), train_cfg['num_train_trajs']
    )
    if move_all_to_device:
        if device.type == "cuda":
            free, total = torch.cuda.mem_get_info()
            print(f"[train] CUDA free/total before dataset move: {free / 1024 ** 3:.2f} / {total / 1024 ** 3:.2f} GB")

        print(f"[train] Moving all trajectories to {device} ...")
        list_of_trajs = move_any_to_device(list_of_trajs, device, non_blocking=False)

        if device.type == "cuda":
            torch.cuda.synchronize()
            free, total = torch.cuda.mem_get_info()
            print(f"[train] CUDA free/total after dataset move:  {free / 1024 ** 3:.2f} / {total / 1024 ** 3:.2f} GB")

    if move_all_to_device:
        if num_workers != 0:
            print("[train] move_all_to_device=True -> forcing num_workers=0")
        num_workers = 0
        pin_memory = False  

    # Build dataset from these trajectories
    dataset = DefPlateDataset(list_of_trajs, world_pos_idxs=feat_idx.world_pos, velocity_idxs=feat_idx.velocity)
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
    model = (EGNN_DefPlate(feat_idx.dim_in, DIM_OUT_VEL, DIM_OUT_STRESS, model_hyperparams, model_cfg['adj_norm'])
             .to(device))

    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['adam_weight_decay'],
        fused=(device.type == "cuda"))
    scheduler = ExponentialLR(optimizer, gamma=train_cfg['gamma_lr_scheduler'])
    amp_enabled = bool(train_cfg.get('amp'))
    scaler = GradScaler(enabled=amp_enabled)

    # Training
    print_training_config(train_cfg, train_loader)
    history = TrainingHistory.create_empty()
    start_time = time.time()

    for epoch in range(train_cfg['epochs']):
        # Train
        train_loss, train_vel, train_stress, grad_norm = _train_one_epoch(model, train_loader, optimizer, device,
                                                                          feat_idx.velocity, feat_idx.stress,
                                                                          amp_enabled, scaler, move_all_to_device)

        # Validate
        val_loss, val_vel, val_stress = _validate_one_epoch(model, test_loader, device, feat_idx.velocity,
                                                            feat_idx.stress, amp_enabled)

        # Record history
        history.train_losses.append(train_loss)
        history.val_losses.append(val_loss)
        history.train_vel_losses.append(train_vel)
        history.train_stress_losses.append(train_stress)
        history.test_vel_losses.append(val_vel)
        history.test_stress_losses.append(val_stress)
        history.grad_norms.append(grad_norm)

        scheduler.step()

        tqdm.write(f"[Train] [Epoch {epoch:03d}] "
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
    parser = argparse.ArgumentParser(description='Train EGNN model')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to config YAML file (default: config.yaml)')
    parser.add_argument('--cuda', action='store_true', default=False,
                        help='Use CUDA if available')
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers (default: 4)')
    parser.add_argument('--pin-memory', action='store_true', default=True,
                        help='Pin memory for faster GPU transfer (default: True)')
    args = parser.parse_args()
    
    device = get_device(args.cuda)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # PyTorch 2.x:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    
    config_file = args.config if args.config else os.path.join(os.path.dirname(__file__), "config.yaml")
    model, test_loader, history, feat_idx, plots_dir = train_egnn(device, args.num_workers, args.pin_memory, config_path=config_file)
    run_final_evaluation(model, test_loader, device, history, feat_idx.velocity, feat_idx.stress, plots_dir,
                         config_path=config_file)
