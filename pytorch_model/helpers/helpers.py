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

def format_training_time(seconds: float) -> str:
    """Format training time as hours, minutes, seconds."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"

def setup_paths(train_cfg: dict) -> Tuple[str, str]:
    """Set up checkpoint and plots directory paths."""
    preprocessed_data_path = train_cfg['datapath']
    add_world_edges = train_cfg['add_world_edges']
    base_name = preprocessed_data_path.rsplit("/", 1)[0]

    checkpoint_path = f"{train_cfg['model_path']}model_{base_name}_{add_world_edges}"
    plots_dir = os.path.join(
        os.path.dirname(__file__),
        f"{train_cfg['model_path']}plots_{base_name}_{add_world_edges}"
    )
    return checkpoint_path, plots_dir

def create_model_hyperparams(model_cfg: dict):
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

def get_device() -> torch.device:
    """Determine the best available device."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def get_feature_indices(include_mesh_pos: bool) -> FeatureIndices:
    """Get feature indices based on whether mesh positions are included."""
    if include_mesh_pos:
        # mesh_pos(3) + world_pos(3) + node_type(2) + vel(3) + stress(1) + kinematic_vel_tp1(3)
        return FeatureIndices(mesh_pos = slice(0,3), world_pos=slice(3, 6), velocity=slice(8, 11), stress=slice(11, 12),
                              dim_in=12 + 3, nodetype=slice(6,8))
    else:
        # world_pos(3) + node_type(2) + vel(3) + stress(1) + kinematic_vel_tp1(3)
        return FeatureIndices(world_pos=slice(0, 3), velocity=slice(5, 8), stress=slice(8, 9), dim_in=9 + 3,
                              nodetype=slice(6,8), mesh_pos=None)


def load_trajectories(data_path: str, num_train_trajs: Optional[int] = None) -> list:
    """Load preprocessed trajectories from disk."""
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {data_path}\n"
            f"Please run 'python preprocess_data.py' first to generate the preprocessed data."
        )

    list_of_trajs = torch.load(data_path)
    print(f"\t Loaded {len(list_of_trajs)} preprocessed trajectories")

    if num_train_trajs is not None and num_train_trajs < len(list_of_trajs):
        list_of_trajs = list_of_trajs[:num_train_trajs]
        print(f"\t Using first {num_train_trajs} trajectories")

    return list_of_trajs

def print_overfit_samples(loader):
    """
    Print the samples being used for overfitting.
    Args:
        loader: DataLoader
            data loader
        """
    batch = next(iter(loader))
    traj_ids, time_indices = batch[8], batch[9]
    print("Overfitting on the following (traj_id, time_idx) pairs:")
    for i, (tr, ti) in enumerate(zip(traj_ids, time_indices)):
        print(f"  sample {i:02d}: traj_id={int(tr)}, t={int(ti)}")

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