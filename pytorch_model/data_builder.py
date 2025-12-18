import json
import torch
import numpy as np
from tfrecord.reader import tfrecord_loader
import os
from helpers.helpers import get_feature_indices, load_config
from helpers.helpers import print_debug_nodetype, print_debug_shapes_dataloader
from data.decode_tfrecord_utils import cast_trajectory_from_record
from data.add_world_edges import add_w_edges
from model_gunet.helpers_models import get_adj_norm_fn
from types import SimpleNamespace
import time

NORMAL_NODE_OH = [0, 0]  # value 0 (NORMAL)
NORMAL_NODE = 0
SPHERE_NODE_OH = [1, 0]  # value 1 (SPHERE)
SPHERE_NODE = 1
BOUNDARY_NODE_OH = [0, 1]  # value 3 (BOUNDARY)
BOUNDARY_NODE = 3
VELOCITY_MEAN = 0.0


def build_edges_from_cells(mesh_cells):
    """
    Receives the set of all mesh cell, each mesh cell is made by 4 points
    0. Declares an empty set to not count edge duplicates
    1. for all cells: convert to standard python list with ints and unpack them in 4 variables
    2. It sorts edge list and then converts to a torch tensor

    Args:
        mesh_cells:
            collection of mesh cells

    :return edge_list: torch.tensor
        torch tensor with shape (#edges, 2) since each edge i-->j is (i, j)
    """
    edge_set = set()
    edge_indices = [(0, 1), (0, 2), (0, 3),
                    (1, 2), (1, 3),
                    (2, 3)]
    for c in mesh_cells:
        # unpack and repack
        i0, i1, i2, i3 = map(int, c.tolist())
        verts = [i0, i1, i2, i3]
        # for all
        for a, b in edge_indices:
            u, v = verts[a], verts[b]
            if u != v:
                edge_set.add((u, v))
                edge_set.add((v, u))

    edge_list = sorted(edge_set)
    return torch.tensor(edge_list, dtype=torch.long)


def build_velocity(world_pos, mode):
    """
    Build velocity array from world positions.

    Args:
        world_pos: np.ndarray
            World positions with shape (T, N, 3)
        mode: str
            'actuator' | 'normal'
    :return vel: np.ndarray
        Velocity array with shape (T, N, 3)
    """
    if mode not in ["normal", "actuator"]:
        raise ValueError(f"Unkown mode = {mode}")

    time_step_dim, number_of_nodes, _ = world_pos.shape
    vel = np.zeros((time_step_dim, number_of_nodes, 3), dtype=np.float32)

    if mode == "normal":
        for t in range(1, time_step_dim):
            vel[t] = world_pos[t] - world_pos[t - 1]
    elif mode == "actuator":
        for t in range(1, time_step_dim - 1):
            vel[t] = world_pos[t+1] - world_pos[t]

    return vel

def build_onehot_nodetype(node_type):
    """
    Convert node type to one-hot encoding.

    Args:
        node_type: np.ndarray
            Node type array with shape (N, 1)
        node_type_onehot: np.ndarray
            One-hot encoded node type with shape (N, 2)
        node_type_raw: np.ndarray
            Copy of original node type
    """
    lookup = np.array([NORMAL_NODE_OH, SPHERE_NODE_OH, [0, 0], BOUNDARY_NODE_OH])
    node_type_raw = node_type.copy()
    node_type_idx = node_type_raw.squeeze(-1)
    node_type_onehot = lookup[node_type_idx]
    return node_type_onehot, node_type_raw


def build_feature_sequence(world_pos, vel, stress, node_type_onehot, mesh_pos,
                           include_mesh_pos, norm_method):
    """
    Build feature sequence for a trajectory.
    Feature layout: [mesh_pos, pos_x, pos_y, pos_z, node_type, vel_x, vel_y, vel_z, stress]

    Args:
        world_pos: np.ndarray
            World positions (T, N, 3)
        vel: np.ndarray
            Velocities (T, N, 3)
        stress: np.ndarray
            Stress values (T, N, 1)
        node_type_onehot: np.ndarray
            One-hot node types (N, 2)
        mesh_pos: np.ndarray or None
            Mesh positions (N, 3)
        include_mesh_pos: bool
            Whether to include mesh positions
        norm_method: str
            Normalization method ('centroid' or 'standard')

    :return X_feat: torch.Tensor
        Feature tensor with shape (T, N, F)
    """
    time_step_dim = world_pos.shape[0]
    feats_list = []
    node_type_floatcast = node_type_onehot.astype(np.float32)

    for t in range(time_step_dim):
        if include_mesh_pos:
            feats_t = np.concatenate([mesh_pos, world_pos[t], node_type_floatcast, vel[t], stress[t]],
                                     axis=-1)
        else:
            feats_t = np.concatenate([world_pos[t], node_type_floatcast, vel[t], stress[t]], axis=-1)

        feats_list.append(feats_t)

    X_feat = torch.tensor(np.stack(feats_list, axis=0), dtype=torch.float32)
    return X_feat


def build_adjacency_matrix(mesh_cells, number_of_nodes):
    """
    Build adjacency matrix from mesh cells.

    Args:
        mesh_cells: np.ndarray
            Mesh cells (C, 4)
        number_of_nodes: int
            Number of nodes
        A: torch.Tensor
            Adjacency matrix (N, N)
    """
    edge_index = build_edges_from_cells(mesh_cells)
    A = torch.zeros((number_of_nodes, number_of_nodes), dtype=torch.float32)
    for e in edge_index:
        A[e[0], e[1]] = 1.0
    return A


def compute_global_mean(list_of_trajs):
    """
    Compute global mean across all trajectories.

    Args:
        list_of_trajs: List
            List of trajectory dicts
        mean: torch.Tensor
            Global mean

    :return element_num: int
        Total number of elements
    """
    sum_elements = 0
    element_num = 0
    for traj in list_of_trajs:
        X_feat = traj['X_seq_norm']
        sum_elements = sum_elements + X_feat.sum(dim=(0, 1))
        element_num = element_num + X_feat.shape[0] * X_feat.shape[1]
    mean = sum_elements / element_num
    return mean, element_num


def compute_standard_normalization(list_of_trajs, mean, element_num, feat_idx, include_mesh_pos):
    """
    Compute normalization statistics using standard method.

    Args:
        list_of_trajs: List
            List of trajectory dicts
        mean: torch.Tensor
            Global mean
        element_num: int
            Total number of elements
        feat_idx: object
            Feature indices
        include_mesh_pos: bool
            Whether mesh positions are included
        mean: torch.Tensor
            Adjusted mean

    :return std_dev: torch.Tensor
        Standard deviation
    """
    # World position
    shared_mean_pos = mean[feat_idx.world_pos].mean()
    mean[feat_idx.world_pos] = shared_mean_pos
    # Velocity
    shared_mean_vel = mean[feat_idx.velocity].mean()
    mean[feat_idx.velocity] = shared_mean_vel

    if include_mesh_pos:
        shared_mean_mesh_pos = mean[feat_idx.mesh_pos].mean()
        mean[feat_idx.mesh_pos] = shared_mean_mesh_pos

    std_acc = torch.zeros_like(mean)
    for traj in list_of_trajs:
        X = traj['X_seq_norm']
        std_acc += ((X - mean.view(1, 1, -1)) ** 2).sum(dim=(0, 1))

    std_dev = torch.sqrt(std_acc / (element_num - 1))

    max_std_pos = std_dev[feat_idx.world_pos].max()
    std_dev[feat_idx.world_pos] = max_std_pos
    max_std_vel = std_dev[feat_idx.velocity].max()
    std_dev[feat_idx.velocity] = max_std_vel

    if include_mesh_pos:
        max_std_mesh_pos = std_dev[feat_idx.mesh_pos].max()
        std_dev[feat_idx.mesh_pos] = max_std_mesh_pos

    return mean, std_dev


def compute_row_normalization(list_of_trajs, mean, element_num):
    """
    Compute normalization statistics component-wise (row-wise) for all features.
    Unlike standard normalization, this does NOT average stats across x,y,z coordinates.

    Args:
        list_of_trajs: List
            List of trajectory dicts
        mean: torch.Tensor
            Global mean (calculated component-wise by compute_global_mean)
        element_num: int
            Total number of elements

    :return (mean, stdev)
        mean: torch.Tensor
            The original component-wise mean
        std_dev: torch.Tensor
            Component-wise standard deviation
    """
    std_acc = torch.zeros_like(mean)

    # Broadcast mean to (1, 1, F) for subtraction
    mean_b = mean.view(1, 1, -1)

    for traj in list_of_trajs:
        X = traj['X_seq_norm']
        # Sum squared differences per component: Sum over T(0) and N(1), keep F
        std_acc += ((X - mean_b) ** 2).sum(dim=(0, 1))

    # Calculate standard deviation
    std_dev = torch.sqrt(std_acc / (element_num - 1))

    # If a feature is constant (e.g., specific node types or 2D constraints), we avoid division by 0
    std_dev[std_dev < 1e-8] = 1.0

    return mean, std_dev

def apply_normalization(list_of_trajs, mean, std_dev):
    """
    Apply normalization to all trajectories.

    Args:
        list_of_trajs: List
            List of trajectory dicts
        mean: torch.Tensor
            Mean for normalization
        std_dev: torch.Tensor
            Standard deviation for normalization
    """
    # Broadcastable shapes
    mean_b = mean.view(1, 1, -1)
    std_b = std_dev.view(1, 1, -1)

    for traj in list_of_trajs:
        traj['mean'] = mean_b
        traj['std'] = std_b
        X = traj['X_seq_norm']
        X_seq_norm = (X - mean_b) / std_b
        traj['X_seq_norm'] = X_seq_norm


def process_single_trajectory(traj, include_mesh_pos, norm_method, idx, add_world_edges_dict, a_time_var, adj_norm_fn):
    """
    Process a single trajectory: decode, build features, and create trajectory dict.

    Args:
        traj: dict
            Decoded trajectory from TFRecord
        include_mesh_pos: bool
            Whether to include mesh positions
        norm_method: str
            Normalization method
        idx: int
            Debug index

    :return
        dict_traj: dict
            Processed trajectory dict
        X_feat: torch.Tensor
            Feature tensor for accumulating statistics
    """
    world_pos = traj["world_pos"]  # (T,N,3)
    stress = traj["stress"]  # (T,N,1)
    node_type = traj["node_type"]  # (N,1)
    mesh_cells = traj["cells"]  # (C,4)
    mesh_pos = None
    if include_mesh_pos:
        mesh_pos = traj["mesh_pos"]
    # print_debug_shapes_dataloader(node_type, idx, mesh_pos, traj, include_mesh_pos, mesh_cells, stress, world_pos)

    time_step_dim, number_of_nodes, _ = world_pos.shape

    # Build velocity
    vel_normal = build_velocity(world_pos, mode="normal")
    vel_actuator_tp1 = build_velocity(world_pos, mode="actuator")
    actuator_mask = (node_type == SPHERE_NODE).reshape(-1)  # Shape (N,)
    vel_normal[:, actuator_mask, :] = vel_actuator_tp1[:, actuator_mask, :]

    # One hot node type
    node_type_onehot, node_type_raw = build_onehot_nodetype(node_type)
    print_debug_nodetype(idx, node_type)

    # Build feature sequence Feature layout: [mesh_pos, pos_x, pos_y, pos_z, node_type, vel_x, vel_y, vel_z, stress]
    X_feat = build_feature_sequence(world_pos, vel_normal, stress, node_type_onehot, mesh_pos,
                                    include_mesh_pos, norm_method)

    # Build adjacency matrix from set
    A = build_adjacency_matrix(mesh_cells, number_of_nodes)

    edge_config = SimpleNamespace(
        add_world_edges=add_world_edges_dict["add_world_edges"],  # Options: 'radius', 'neighbours', 'None'
        radius=add_world_edges_dict["radius_world_edge"],  # Adjust radius
        k_neighb=add_world_edges_dict["k_neighb"]  # Adjust k neighbors
    )
    # time_start = time.time()
    
    if a_time_var:
        # Time-varying adjacency: compute per time step
        A_dynamic_list = []
        dynamic_edges_list = []
        for t in range(time_step_dim):
            pos_t = torch.tensor(world_pos[t], dtype=torch.float32)
            node_type_t = torch.tensor(node_type_raw.squeeze(), dtype=torch.long)
            A_dynamic_t, dynamic_edges_t = add_w_edges(edge_config, A, node_type_t, pos_t)
            A_dynamic_list.append(A_dynamic_t)
            dynamic_edges_list.append(dynamic_edges_t)
        A_out = torch.stack(A_dynamic_list, dim=0)  # [T, N, N]
        world_edges_out = dynamic_edges_list
    else:
        # Static adjacency: compute once at t=0
        pos_t = torch.tensor(world_pos[0], dtype=torch.float32)
        node_type_t = torch.tensor(node_type_raw.squeeze(), dtype=torch.long)
        A_static, dynamic_edges_static = add_w_edges(edge_config, A, node_type_t, pos_t)
        A_out = A_static  # [N, N]
        world_edges_out = dynamic_edges_static  # single tensor for compatibility

    # Normalize adjacency (static [N,N] or time-varying [T,N,N])
    A_out = adj_norm_fn(A_out.to(torch.float32))

    # compute_duration = time.time() - time_start
    # print(f"[process_single_trajectory] Added world edges in {compute_duration:.4f}s")
    
    # ensure cells and node_type are tensors, passing them to plot border and sphere separately (not predicted)
    cells_tensor = torch.tensor(mesh_cells, dtype=torch.long)
    node_type_tensor = torch.tensor(node_type_raw.squeeze(-1), dtype=torch.long)
    dict_traj = {"A": A_out, "X_seq_norm": X_feat, "mean": 0, "std": 0, "cells": cells_tensor,
                 "node_type": node_type_tensor, "world_edge_index": world_edges_out}

    return dict_traj, X_feat


def load_all_trajectories(dataconfig):
    """
    Load up to `max_trajs` trajectories from TFRecord.

    Args:
        tfrecord_path: str
            path of the tfrecord files
        meta_path: str
            path of the meta.json file
        max_trajs: int
            maximum number of trajectories to load

    :return list_of_trajs: List
        list of dicts where each dict contains:
          - "A"          : [N,N] adjacency matrix (torch.float32)
          - "X_seq_norm" : [T,N,F] normalized features (torch.float32)
          - "mean"       : [1,1,F] mean for denorm
          - "std"        : [1,1,F] std for denorm
          - "cells"      : [C,4] connectivity (torch.long)
    """

    include_mesh_pos = dataconfig['include_mesh_pos']
    norm_method = dataconfig['normalization_method']
    adj_norm = dataconfig['adj_norm']
    tfrecord_path = dataconfig['tfrecord_path']
    meta_path = dataconfig['meta_path']
    max_trajs = dataconfig['max_trajs']

    add_world_edges_dict = {'add_world_edges': dataconfig['add_world_edges'],
                            'radius_world_edge': dataconfig['radius_world_edge'],
                            'k_neighb':dataconfig['k_neighb']}
    a_time_var = dataconfig.get('a_time_var')

    print(dataconfig['add_world_edges'])
    if dataconfig['add_world_edges'] not in ['radius', 'neighbours', 'None']:
        raise ValueError(f"add_world_edges == {dataconfig['add_world_edges']} not supported")
    if norm_method not in ['centroid', 'standard', 'row']:
        raise ValueError(f"norm_method == {norm_method} not supported")
    if adj_norm not in ['row', 'sym']:
        raise ValueError(f"adj_norm == {adj_norm} not supported (expected 'row' or 'sym')")

    adj_norm_fn = get_adj_norm_fn(adj_norm)

    feat_idx = get_feature_indices(include_mesh_pos)

    # Load meta.json for decoding
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # TFRecord loader
    loader = tfrecord_loader(tfrecord_path, index_path=None)
    list_of_trajs = []
    idx = 0  # debug idx

    # Iterate through trajectories
    for traj_idx, record in enumerate(loader):
        print(f"processing trajectory {traj_idx}")
        # Stop if we reached max_trajs
        if max_trajs is not None and traj_idx >= max_trajs:
            print("[load_all_trajectories] Reached wanted number of trajectories")
            break

        traj = cast_trajectory_from_record(record, meta)
        dict_traj, X_feat = process_single_trajectory(
            traj, include_mesh_pos, norm_method, idx, add_world_edges_dict, a_time_var, adj_norm_fn
        )
        list_of_trajs.append(dict_traj)

    mean, element_num = compute_global_mean(list_of_trajs)

    if norm_method == "standard":
        mean, std_dev = compute_standard_normalization(list_of_trajs, mean, element_num,
                                                       feat_idx, include_mesh_pos)
    elif norm_method == "row":
        mean, std_dev = compute_row_normalization(list_of_trajs, mean, element_num)
    else:
        raise ValueError(f"[load_all_trajectories] norm_method={norm_method}")

    apply_normalization(list_of_trajs, mean, std_dev)

    print(f"\nLoaded {len(list_of_trajs)} trajectories.")
    return list_of_trajs


def load_all_trajectories_with_precomputed_stats(dataconfig, mean, std_dev):
    """
    Load trajectories from TFRecord and apply precomputed normalization statistics.

    This is intended for validation/test splits: features are built exactly like train
    (including recomputing world edges), but mean/std are NOT refit on this split.

    Args:
        dataconfig: dict
            Same dataconfig used for training preprocessing, except `tfrecord_path` points
            to the split to load (e.g., valid.tfrecord).
        mean: torch.Tensor
            Train-fitted mean (shape [F] or [1,1,F]).
        std_dev: torch.Tensor
            Train-fitted std (shape [F] or [1,1,F]).
    """
    include_mesh_pos = dataconfig['include_mesh_pos']
    norm_method = dataconfig['normalization_method']
    adj_norm = dataconfig['adj_norm']
    tfrecord_path = dataconfig['tfrecord_path']
    meta_path = dataconfig['meta_path']
    max_trajs = dataconfig['max_trajs']

    add_world_edges_dict = {
        'add_world_edges': dataconfig['add_world_edges'],
        'radius_world_edge': dataconfig['radius_world_edge'],
        'k_neighb': dataconfig['k_neighb']
    }
    a_time_var = dataconfig.get('a_time_var')

    if dataconfig['add_world_edges'] not in ['radius', 'neighbours', 'k_neighb', 'None']:
        raise ValueError(f"add_world_edges == {dataconfig['add_world_edges']} not supported")
    if norm_method not in ['centroid', 'standard', 'row']:
        raise ValueError(f"norm_method == {norm_method} not supported")
    if adj_norm not in ['row', 'sym']:
        raise ValueError(f"adj_norm == {adj_norm} not supported (expected 'row' or 'sym')")

    adj_norm_fn = get_adj_norm_fn(adj_norm)

    # Load meta.json for decoding
    with open(meta_path, "r") as f:
        meta = json.load(f)

    loader = tfrecord_loader(tfrecord_path, index_path=None)
    list_of_trajs = []
    idx = 0  # debug idx

    for traj_idx, record in enumerate(loader):
        print(f"processing trajectory {traj_idx}")
        if max_trajs is not None and traj_idx >= max_trajs:
            print("[load_all_trajectories_with_precomputed_stats] Reached wanted number of trajectories")
            break

        traj = cast_trajectory_from_record(record, meta)
        dict_traj, _ = process_single_trajectory(
            traj, include_mesh_pos, norm_method, idx, add_world_edges_dict, a_time_var, adj_norm_fn
        )
        list_of_trajs.append(dict_traj)

    # Apply *train-fitted* normalization (broadcasting handled in apply_normalization)
    apply_normalization(list_of_trajs, mean, std_dev)

    print(f"\nLoaded {len(list_of_trajs)} trajectories (with precomputed normalization).")
    return list_of_trajs


if __name__ == "__main__":
    dataconfig_path = os.path.join(os.path.dirname(__file__), "dataconfig.yaml")
    dataconfig = load_config(dataconfig_path)
    list_of_trajs = load_all_trajectories(dataconfig)