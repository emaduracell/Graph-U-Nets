import yaml
import torch
from dataclasses import dataclass
import os


@dataclass
class FeatureIndices:
    """
    Container for feature slice indices.

    Args:
        world_pos: slice for world position indices.
        velocity: slice for velocity indices.
        stress: slice for stress indices.
        dim_in: int representing the input dimension size.
        mesh_pos: slice for mesh position indices or None.
        nodetype: slice for node type indices.
    """
    world_pos: slice
    velocity: slice
    stress: slice
    dim_in: int
    mesh_pos: slice | None
    nodetype: slice


def load_config(config_path):
    """
    Load model and training configuration from YAML file.

    Args:
        config_path: str, path to the configuration YAML file.

    Returns:
        dict: The loaded configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def format_training_time(seconds):
    """
    Format training time as hours, minutes, seconds.

    Args:
        seconds: float, the duration in seconds.

    Returns:
        str: Formatted time string.
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes}m {secs}s"


def setup_paths(train_cfg):
    """
    Creates:
      model_out/<dataset_name>/model.pt
      model_out/<dataset_name>/plots/
    where <dataset_name> is the last folder of train_cfg['datapath']

    Args:
        train_cfg: dict, training configuration containing paths.

    Returns:
        tuple: A tuple containing the checkpoint path and plots directory path.
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
    """
    Create model hyperparameters object from config.

    Args:
        model_cfg: dict, configuration dictionary for the model.

    Returns:
        object: An object with attributes set from the config.
    """
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
            data loader containing the training data
        train_cfg: Dict
            dictionary containing train configuration parameters
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


def get_device(cuda):
    """
    Determine the best available device.

    Args:
        cuda: bool, flag indicating whether to try using CUDA.

    Returns:
        torch.device: The selected device (CUDA, MPS, or CPU).
    """
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


def get_device_pyg(cuda):
    """
    Determine the best available device.

    Args:
        cuda: bool, flag indicating whether to try using CUDA.

    Returns:
        torch.device: The selected device (CUDA or CPU).
    """
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
        if torch.cuda.is_available():
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
    """
    Get feature indices based on whether mesh positions are included.

    Args:
        include_mesh_pos: bool, flag to determine if mesh positions should be included.

    Returns:
        FeatureIndices: Object containing slice indices for features.
    """
    if include_mesh_pos:
        # mesh_pos(3) + world_pos(3) + node_type(2) + vel(3) + stress(1) + kinematic_vel_tp1(3)
        return FeatureIndices(mesh_pos=slice(0, 3), world_pos=slice(3, 6), velocity=slice(8, 11), stress=slice(11, 12),
                              dim_in=12, nodetype=slice(6, 8))
    else:
        # world_pos(3) + node_type(2) + vel(3) + stress(1) + kinematic_vel_tp1(3)
        return FeatureIndices(world_pos=slice(0, 3), velocity=slice(5, 8), stress=slice(8, 9), dim_in=9,
                              nodetype=slice(6, 8), mesh_pos=None)


def load_trajectories_preprocessed(data_path, num_train_trajs):
    """
    Load preprocessed trajectories from disk.

    Args:
        data_path: str
            Path to the saved trajectory data file.
        num_train_trajs: int or None
            Maximum number of trajectories to load.
    :return: List of loaded trajectory tensors.
    """
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
    """
    Print information about samples in the loader for overfitting diagnosis.

    Args:
        loader: DataLoader
            The data loader containing the batch.
    """
    batch = next(iter(loader))
    adj, X_t, X_tp1, mean, std, cells, node_types, traj_ids, time_indices = batch

    print("Overfitting on the following (traj_id, time_idx) pairs:")
    for i, (tr, ti) in enumerate(zip(traj_ids, time_indices)):
        print(f"  sample {i:02d}: traj_id={tr}, t={ti}")


def print_debug_shapes_dataloader(node_type, idx, mesh_pos, traj, include_mesh_pos, mesh_cells, stress, world_pos):
    """
    Print debug information regarding tensor shapes and types.

    Args:
        node_type: tensor or array
            Data representing node types.
        idx: int
            Current iteration index.
        mesh_pos: tensor or array
            Data representing mesh positions.
        traj: object
            The trajectory object being processed.
        include_mesh_pos: bool
            Flag indicating if mesh position is included.
        mesh_cells: tensor or array
            Data representing mesh cells.
        stress: tensor or array
            Data representing stress values.
        world_pos: tensor or array
            Data representing world positions.
    """

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
    """
    Print debug information specifically for node types.

    Args:
        idx: int
            Current iteration index.
        node_type: tensor or array
            Data representing node types.
    """
    # Debug
    if idx == 1 or idx == 2:
        print(
            f"[data_loader] node_type: \n \t type(node_type) = {type(node_type)} \n \t type(node_type[0])={type(node_type[0])}, "
            f"\n \t type(node_type[0][0])={type(node_type[0][0])}) \n \t len(node_type)={len(node_type)} "
            f"\n \t len(node_type[0])={len(node_type[0])}")


def tensor_bytes(x):
    """
    Calculate the total size in bytes of the input tensor or collection.

    Args:
        x: tensor, list, dict or tuple
            The input object to calculate size for.
    """
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
        obj: tensor, list, dict, or tuple
            The object to move to the device.
        device: torch.device
            The target device.
        non_blocking: bool
            parameter set to True to avoid making training slower

    :return: object to device
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