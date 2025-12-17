import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
from data.defplate_dataset import DefPlateDataset, collate_unet, collate_block_diagonal
from model.gunet_deforming_plate import GraphUNet_DefPlate
from torch.optim.lr_scheduler import ExponentialLR
import time
from typing import List, Tuple
from dataclasses import dataclass
from tqdm import tqdm
from helpers.evaluation_helper import run_final_evaluation
from helpers.helpers import (format_training_time, create_model_hyperparams, load_config, load_trajectories_preprocessed,
                             print_training_config, setup_paths, get_feature_indices, get_device, print_overfit_samples,
                             move_any_to_device)
# from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler

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
    Create train/val loaders with block-diagonal batching plus a list-of-graphs
    eval loader for plotting.
    """
    total = len(dataset)
    perm = torch.randperm(total)
    split = int(0.8 * total)

    train_idx = perm[:split]
    val_idx = perm[split:]

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_block_diagonal,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_block_diagonal,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    eval_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_unet,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, eval_loader

def _create_overfit_dataloader(dataset, overfit_traj_id, overfit_time_idx_list):
    """
    Create dataloaders for overfitting on specific samples.

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
        raise ValueError(
            f"No samples found matching overfit criteria: "
            f"traj_id={overfit_traj_id}, time_idx={overfit_time_idx_list}"
        )

    overfit_set = Subset(dataset, overfit_indices)
    # Block-diagonal loader for training/validation
    train_val_loader = DataLoader(
        overfit_set, batch_size=len(overfit_indices), shuffle=False, collate_fn=collate_block_diagonal
    )
    # List-of-graphs loader for evaluation/plots
    eval_loader = DataLoader(
        overfit_set, batch_size=len(overfit_indices), shuffle=False, collate_fn=collate_unet
    )

    print(f"\nOverfitting on trajectory {overfit_traj_id} with {len(overfit_indices)} time steps")
    print_overfit_samples(eval_loader)

    return train_val_loader, eval_loader


def compute_loss_vectorized(preds, targets, nodetypes, velocity_idxs, stress_idxs):
    """
    Vectorized Huber loss over a disjoint-union batch.

    Args:
        preds: Tensor [B*N, F_out]
        targets: Tensor [B*N, F_in]
        nodetypes: Tensor [B*N]
    """
    vel_mask = (nodetypes == NORMAL_NODE)
    stress_mask = (nodetypes == NORMAL_NODE) | (nodetypes == BOUNDARY_NODE)

    target_vel = targets[:, velocity_idxs]
    target_stress = targets[:, stress_idxs]
    pred_vel = preds[:, :3]
    pred_stress = preds[:, 3:4]

    vel_loss = torch.tensor(0.0, device=preds.device)
    stress_loss = torch.tensor(0.0, device=preds.device)

    if vel_mask.any():
        vel_loss = F.huber_loss(pred_vel[vel_mask], target_vel[vel_mask])

    if stress_mask.any():
        stress_loss = F.huber_loss(pred_stress[stress_mask], target_stress[stress_mask])

    return vel_loss + stress_loss, vel_loss, stress_loss


def _get_grad_norm(model):
    """Get gradient norm of current batch (L2)."""
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


@torch.no_grad()
def _validate_one_epoch(model, val_loader, device, velocity_idxs, stress_idxs, amp_enabled: bool):
    """Run one validation epoch with block-diagonal batches."""
    model.eval()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0

    for batch in tqdm(val_loader, desc="Val", leave=False):
        batch_adj, batch_xt, batch_xtp1, batch_nt = batch

        batch_adj = batch_adj.to(device, non_blocking=True)
        batch_xt = batch_xt.to(device, non_blocking=True)
        batch_xtp1 = batch_xtp1.to(device, non_blocking=True)
        batch_nt = batch_nt.to(device, non_blocking=True)

        if device.type == 'cuda':
            with autocast(device_type=device.type, enabled=amp_enabled):
                preds = model(batch_adj, batch_xt, batch_xtp1, batch_nt)
        else:
            preds = model(batch_adj, batch_xt, batch_xtp1, batch_nt)

        batch_loss, vel_loss, stress_loss = compute_loss_vectorized(
            preds, batch_xtp1, batch_nt, velocity_idxs, stress_idxs
        )

        total_loss += batch_loss.detach()
        total_vel_loss += vel_loss.detach()
        total_stress_loss += stress_loss.detach()

    n = len(val_loader)
    return total_loss.item() / n, total_vel_loss.item() / n, total_stress_loss.item() / n


def _train_one_epoch(model, train_loader, optimizer, device, velocity_idxs, stress_idxs, amp_enabled, scaler,
                     move_all_to_device):
    """Run one training epoch using block-diagonal batches."""
    model.train()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    total_grad_norm = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Train", leave=False):
        batch_adj, batch_xt, batch_xtp1, batch_nt = batch

        if not move_all_to_device and batch_adj.device != device:
            batch_adj = batch_adj.to(device)
            batch_xt = batch_xt.to(device)
            batch_xtp1 = batch_xtp1.to(device)
            batch_nt = batch_nt.to(device)

        optimizer.zero_grad(set_to_none=True)

        if device.type == 'cuda':
            with autocast(device_type=device.type, enabled=amp_enabled):
                preds = model(batch_adj, batch_xt, batch_xtp1, batch_nt)
                batch_loss, vel_loss, stress_loss = compute_loss_vectorized(
                    preds, batch_xtp1, batch_nt, velocity_idxs, stress_idxs
                )
        else:
            preds = model(batch_adj, batch_xt, batch_xtp1, batch_nt)
            batch_loss, vel_loss, stress_loss = compute_loss_vectorized(
                preds, batch_xtp1, batch_nt, velocity_idxs, stress_idxs
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

def train_gunet(device, num_workers, pin_memory):
    """Training loop"""
    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    # Extract model and training parameters
    model_cfg = config['model']
    train_cfg = config['training']
    # Load train config
    # datapath: processed_data/data_standard_True so add preprocessed_train.pt
    checkpoint_path, plots_dir = setup_paths(train_cfg)
    dataconfig = load_config(train_cfg['datapath'] + '/used_dataconfig.yaml')
    include_mesh_pos = dataconfig['include_mesh_pos']
    feat_idx = get_feature_indices(include_mesh_pos)
    torch.manual_seed(train_cfg['random_seed'])
    np.random.seed(train_cfg['random_seed'])
    move_all_to_device = bool(train_cfg.get("move_all_to_device"))

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

    list_of_trajs = load_trajectories_preprocessed(
        train_cfg['datapath'] + "/preprocessed_train.pt", train_cfg['num_train_trajs']
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
        # IMPORTANT: GPU-resident dataset + multi-worker dataloader is a footgun.
        if num_workers != 0:
            print("[train] move_all_to_device=True -> forcing num_workers=0")
        num_workers = 0
        pin_memory = False  # irrelevant / sometimes harmful here

    # Build dataset from these trajectories
    dataset = DefPlateDataset(list_of_trajs, world_pos_idxs=feat_idx.world_pos, velocity_idxs=feat_idx.velocity)
    print(f"Total training pairs (X_t, X_t+1): {len(dataset)}")

    # Create dataloaders based on mode
    if train_cfg['mode'] == "overfit":
        train_loader, eval_loader = _create_overfit_dataloader(
            dataset, train_cfg.get('overfit_traj_id'), train_cfg.get('overfit_time_idx', [])
        )
        val_loader = train_loader
    else:
        train_loader, val_loader, eval_loader = _create_standard_dataloaders(
            dataset, train_cfg['batch_size'], train_cfg['shuffle'], num_workers, pin_memory
        )

    # Build model and optimizer
    model_hyperparams = create_model_hyperparams(model_cfg)
    model = (GraphUNet_DefPlate(feat_idx.dim_in, DIM_OUT_VEL, DIM_OUT_STRESS, model_hyperparams, model_cfg['adj_norm'])
             .to(device))

    optimizer = optim.Adam(
        model.parameters(),
        lr=train_cfg['lr'],
        weight_decay=train_cfg['adam_weight_decay'],
        fused=(device.type == "cuda")
    )
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
        val_loss, val_vel, val_stress = _validate_one_epoch(model, val_loader, device, feat_idx.velocity,
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

    return model, eval_loader, history, feat_idx, plots_dir

if __name__ == "__main__":
    cuda = False
    num_workers = 0
    pin_memory = False
    device = get_device(cuda)
    # TODO: set to false if you have compatibility problems
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    # PyTorch 2.x:
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    model, eval_loader, history, feat_idx, plots_dir = train_gunet(device, num_workers, pin_memory)
    run_final_evaluation(model, eval_loader, device, history, feat_idx.velocity, feat_idx.stress, plots_dir,
                         config_path=os.path.join(os.path.dirname(__file__), "config.yaml"))
