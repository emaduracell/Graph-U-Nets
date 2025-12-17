import yaml
import torch
from dataclasses import dataclass
import os
from typing import List, Tuple, Optional
import numpy as np

@dataclass
class FeatureIndices:
    """Container for feature slice indices."""
    world_pos: slice
    velocity: slice
    stress: slice
    dim_in: int
    mesh_pos: slice | None
    nodetype: slice

def load_config(config_path):
    """Load model and training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def format_training_time(seconds):
    """Format training time as hours, minutes, seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

# def setup_paths(train_cfg):
#     """Set up checkpoint and plots directory paths."""
#     preprocessed_data_path = train_cfg['datapath'] + "/preprocessed_train.pt"
#     dataconfig = load_config(train_cfg['datapath'] + "/used_dataconfig.yaml")
#     base_name = preprocessed_data_path.rsplit("/", 1)[0]
#
#     checkpoint_path = f"{train_cfg['model_path_out']}model_{base_name}/"
#     plots_dir = os.path.join(f"{train_cfg['model_path_out']}model_{base_name}", "plots")
#     return checkpoint_path, plots_dir

import os

def setup_paths(train_cfg):
    """
    Creates:
      model_out/<dataset_name>/model.pt
      model_out/<dataset_name>/plots/
    where <dataset_name> is the last folder of train_cfg['datapath']
    """
    dataset_name = os.path.basename(os.path.normpath(train_cfg["datapath"]))

    out_dir = os.path.join(train_cfg["model_path_out"], dataset_name)
    checkpoint_path = os.path.join(out_dir, "model.pt")
    plots_dir = os.path.join(out_dir, "plots")

    # Ensure directories exist
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    return checkpoint_path, plots_dir


def create_model_hyperparams(model_cfg):
    """Create model hyperparameters object from config."""
    hyperparams = lambda: None
    hyperparams.activation_gnn = model_cfg['activation_gnn']
    hyperparams.activation_mlps_final = model_cfg['activation_mlps_final']
    hyperparams.hid_gnn_layer_dim = model_cfg['hid_gnn_layer_dim']
    hyperparams.hid_mlp_dim = model_cfg['hid_mlp_dim']
    hyperparams.k_pool_ratios = model_cfg['k_pool_ratios']
    hyperparams.dropout_gnn = model_cfg['dropout_gnn']
    hyperparams.dropout_mlps_final = model_cfg['dropout_mlps_final']
    return hyperparams

def print_training_config(train_cfg, train_loader):
    """
    Print training configuration summary.

    Args:
        train_loader: DataLoader
            data loader
        train_cfg: Dict
            dictionary containing train configuration
    """
    print("\n=================================================")
    print("                  TRAINING")
    print("=================================================\n")
    print(f"Epochs: {train_cfg['epochs']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Start learning rate: {train_cfg['lr']}")
    print(f"Mode: {train_cfg['mode']}")
    print(f"Weight decay: {train_cfg['adam_weight_decay']}")
    print(f"Number of trajectories: {train_cfg['num_train_trajs']}")
    print(f"Train loader batches: {len(train_loader)}\n")

def get_device(cuda: bool):
    """Determine the best available device."""
    if cuda:
        if torch.cuda.is_available():
            dev = torch.device("cuda")
            try:
                name = torch.cuda.get_device_name(dev)
                print(f"[get_device] Using CUDA device: {name}")
            except Exception:
                print(f"[get_device] Using CUDA device: {dev}")
        else:
            raise ValueError("CUDA is not available")
    else:
        if torch.backends.mps.is_available():
            dev = torch.device("mps")
            print(f"[get_device] Using device: {dev}")
        elif torch.cuda.is_available():
            dev = torch.device("cuda")
            try:
                name = torch.cuda.get_device_name(dev)
                print(f"[get_device] Using CUDA device: {name}")
            except Exception:
                print(f"[get_device] Using CUDA device: {dev}")
        else:
            dev = torch.device("cpu")
            print(f"[get_device] Using device: {dev}")
    return dev

def get_feature_indices(include_mesh_pos):
    """Get feature indices based on whether mesh positions are included."""
    if include_mesh_pos:
        # mesh_pos(3) + world_pos(3) + node_type(2) + vel(3) + stress(1) + kinematic_vel_tp1(3)
        return FeatureIndices(mesh_pos = slice(0,3), world_pos=slice(3, 6), velocity=slice(8, 11), stress=slice(11, 12),
                              dim_in=12, nodetype=slice(6,8))
    else:
        # world_pos(3) + node_type(2) + vel(3) + stress(1) + kinematic_vel_tp1(3)
        return FeatureIndices(world_pos=slice(0, 3), velocity=slice(5, 8), stress=slice(8, 9), dim_in=9,
                              nodetype=slice(6,8), mesh_pos=None)


def load_trajectories_preprocessed(data_path, num_train_trajs):
    """Load preprocessed trajectories from disk."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}\n"
            f"Please run 'python preprocess_data.py' first to generate the preprocessed data."
        )

    list_of_trajs = torch.load(data_path)
    gb = tensor_bytes(list_of_trajs) / 1024 ** 3
    print(f"[load_trajectories_preprocessed] Tensor payload size: {gb:.2f} GB")
    print(f"\t [load_trajectories_preprocessed] Loaded {len(list_of_trajs)} preprocessed trajectories")

    if num_train_trajs is not None and num_train_trajs < len(list_of_trajs):
        list_of_trajs = list_of_trajs[:num_train_trajs]
        print(f"\t [load_trajectories_preprocessed] Using first {num_train_trajs} trajectories")

    return list_of_trajs

def print_overfit_samples(loader):
    batch = next(iter(loader))
    adj, X_t, X_tp1, mean, std, cells, node_types, traj_ids, time_indices = batch

    print("Overfitting on the following (traj_id, time_idx) pairs:")
    for i, (tr, ti) in enumerate(zip(traj_ids, time_indices)):
        print(f"  sample {i:02d}: traj_id={tr}, t={ti}")


def print_debug_shapes_dataloader(node_type, idx, mesh_pos, traj, include_mesh_pos, mesh_cells, stress, world_pos):
    if idx == 0 or idx == 1 or idx == 2:
        print(f"traj: \n \t type(traj) = {type(traj)}, len={len(traj)}")
        if include_mesh_pos:
            print(f"mesh pos: \n"
                  f"\t type(mesh_pos) = {type(mesh_pos)} \n \t type(mesh_pos[0])={type(mesh_pos[0])}, "
                  f"\n \t shape(mesh_pos) = {mesh_pos.shape} \n \t shape(mesh_pos[0])={type(mesh_pos[0].shape)}"
                  f"\n \t type(mesh_pos[0][0])={type(mesh_pos[0][0])}) \n \t len(mesh_pos)={len(mesh_pos)} "
                  f"\n \t len(mesh_pos[0])={len(mesh_pos[0])}")
        print(f"world pos: \n"
              f"\t type(world_pos) = {type(world_pos)} \n \t type(world_pos[0])={type(world_pos[0])}, "
              f"\n \t type(world_pos[0][0])={type(world_pos[0][0])}) \n \t len(world_pos)={len(world_pos)} "
              f"\n \t len(world_pos[0])={len(world_pos[0])}")
        print(f"stress: \n \t type(stress) = {type(stress)} \n \t type(stress[0])={type(stress[0])}, "
              f"\n \t type(stress[0][0])={type(stress[0][0])}) \n \t type(stress[0][0][0])={type(stress[0][0][0])})"
              f"\n \t len(stress)={len(stress)} \n \t len(stress[0])={len(stress[0])} "
              f"\n \t len(stress[0][0])={len(stress[0][0])}) ")
        print(
            f"node_type: \n \t type(node_type) = {type(node_type)} \n \t type(node_type[0])={type(node_type[0])}, "
            f"\n \t type(node_type[0][0])={type(node_type[0][0])}) \n \t len(node_type)={len(node_type)} "
            f"\n \t len(node_type[0])={len(node_type[0])}")
        print(
            f"mesh_cells \n \t type(mesh_cells) = {type(mesh_cells)} \n \t type(mesh_cells[0])={type(mesh_cells[0])}, "
            f"\n \t type(mesh_cells[0][0])={type(mesh_cells[0][0])}) \n \t len(mesh_cells)={len(mesh_cells)} "
            f"\n \t len(mesh_cells[0])={len(mesh_cells[0])}")
        idx += 1

def print_debug_nodetype(idx, node_type):
    # Debug
    if idx == 1 or idx == 2:
        print(
            f"[data_loader] node_type: \n \t type(node_type) = {type(node_type)} \n \t type(node_type[0])={type(node_type[0])}, "
            f"\n \t type(node_type[0][0])={type(node_type[0][0])}) \n \t len(node_type)={len(node_type)} "
            f"\n \t len(node_type[0])={len(node_type[0])}")

def tensor_bytes(x):
    if torch.is_tensor(x):
        return x.nelement() * x.element_size()
    if isinstance(x, dict):
        return sum(tensor_bytes(v) for v in x.values())
    if isinstance(x, (list, tuple)):
        return sum(tensor_bytes(v) for v in x)
    return 0

def move_any_to_device(obj, device, non_blocking):
    """
    Recursively move tensors inside nested (dict/list/tuple) structures to device.

    Args:
        obj:
        device: torch.device
        non_blocking: bool
    :return:
    """
    if torch.is_tensor(obj):
        if obj.device == device:
            return obj
        return obj.to(device, non_blocking=non_blocking)
    if isinstance(obj, dict):
        return {k: move_any_to_device(v, device, non_blocking=non_blocking) for k, v in obj.items()}
    if isinstance(obj, list):
        return [move_any_to_device(v, device, non_blocking=non_blocking) for v in obj]
    if isinstance(obj, tuple):
        return tuple(move_any_to_device(v, device, non_blocking=non_blocking) for v in obj)
    return obj