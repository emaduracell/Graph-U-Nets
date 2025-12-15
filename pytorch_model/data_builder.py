import json
import torch
import numpy as np
from tfrecord.reader import tfrecord_loader
import os
from helpers.helpers import get_feature_indices, load_config
from pytorch_model.helpers.helpers import print_debug_nodetype, print_debug_shapes_dataloader
from data.decode_tfrecord_utils import cast_trajectory_from_record

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
        for t in range(1, time_step_dim):
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
        if norm_method == "centroid":
            # Compute frame centroid for world_pos
            centroid_world = world_pos[t].mean(axis=0)  # [3]
            centered_world_pos = world_pos[t] - centroid_world

            if include_mesh_pos:
                # Compute frame centroid for mesh_pos (static, but done per frame for consistency if needed, though mesh_pos is static)
                # Actually mesh_pos is static [N, 3], so we compute its centroid once per traj
                centroid_mesh = mesh_pos.mean(axis=0)
                centered_mesh_pos = mesh_pos - centroid_mesh

            # This removes the rigid/global translation component of velocity
            centroid_vel = vel[t].mean(axis=0)  # [3]
            vel_centered = vel[t] - centroid_vel  # [N,3]

            if include_mesh_pos:
                feats_t = np.concatenate([centered_mesh_pos, centered_world_pos, node_type_floatcast,
                                          vel_centered, stress[t]], axis=-1)
            else:
                feats_t = np.concatenate([centered_world_pos, node_type_floatcast, vel_centered,
                                          stress[t]], axis=-1)

        else:
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


def compute_centroid_normalization(list_of_trajs, mean, element_num, feat_idx, include_mesh_pos):
    """
    Compute normalization statistics using centroid method.

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
    # Force velocity mean to 0
    mean[feat_idx.velocity] = VELOCITY_MEAN

    # 2. Compute Global Standard Deviation
    # We need to re-iterate to calculate variance correctly
    accumulated_variance = torch.zeros_like(mean)
    for traj in list_of_trajs:
        X = traj['X_seq_norm']
        # For velocity, since we forced mean=0, this computes sum(v^2), which leads to RMS
        accumulated_variance += ((X - mean.view(1, 1, -1)) ** 2).sum(dim=(0, 1))
    # Standard Deviation (or RMS for velocity)
    std_dev = torch.sqrt(accumulated_variance / (element_num - 1))

    # B. World Position: Isotropic Std across x, y, z
    # Since we centered positions per-frame, the mean is ~0.
    pos_variances = accumulated_variance[feat_idx.world_pos]
    pos_std_isotropic = torch.sqrt(pos_variances.sum() / ((element_num - 1) * 3))
    std_dev[feat_idx.world_pos] = pos_std_isotropic

    if include_mesh_pos:
        # C. Mesh Position: Isotropic Std across x, y, z
        mesh_variances = accumulated_variance[feat_idx.mesh_pos]
        mesh_std_isotropic = torch.sqrt(mesh_variances.sum() / ((element_num - 1) * 3))
        std_dev[feat_idx.mesh_pos] = mesh_std_isotropic

    # 4. Isotropic scaling for Velocity
    # max_std_vel = std_dev[VELOCITY_INDEXES].max()
    # std_dev[VELOCITY_INDEXES] = max_std_vel
    vel_variances = accumulated_variance[feat_idx.world_pos]  # Shape [3]
    # Sum of squared errors for all 3 components / (Total Elements * 3)
    # Note: element_num is N*T. The total count for 3 components is element_num * 3
    vel_rms = torch.sqrt(vel_variances.sum() / ((element_num - 1) * 3))
    std_dev[feat_idx.world_pos] = vel_rms

    # 5. Node Type: keep one-hot (no normalization)
    # mean is already computed, but we force it to 0 and std to 1 for node types
    mean[feat_idx.nodetype] = 0.0
    std_dev[feat_idx.nodetype] = 1.0

    return mean, std_dev


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


def process_single_trajectory(traj, include_mesh_pos, norm_method, idx):
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
    print_debug_shapes_dataloader(node_type, idx, mesh_pos, traj, include_mesh_pos, mesh_cells, stress, world_pos)

    time_step_dim, number_of_nodes, _ = world_pos.shape

    # Build velocity
    vel_normal = build_velocity(world_pos, mode="normal")
    vel_actuator_tp1 = build_velocity(world_pos, mode="actuator")
    # FIXME SEE IF THIS ACTUALLY DOES WHAT IT SHOULD DO
    vel_normal[node_type == SPHERE_NODE] = vel_actuator_tp1[node_type == SPHERE_NODE]

    # One hot node type
    node_type_onehot, node_type_raw = build_onehot_nodetype(node_type)
    print_debug_nodetype(idx, node_type)

    # Build feature sequence Feature layout: [mesh_pos, pos_x, pos_y, pos_z, node_type, vel_x, vel_y, vel_z, stress]
    X_feat = build_feature_sequence(world_pos, vel_normal, stress, node_type_onehot, mesh_pos,
                                    include_mesh_pos, norm_method)

    # Build adjacency matrix from set
    A = build_adjacency_matrix(mesh_cells, number_of_nodes)

    # ensure cells and node_type are tensors, passing them to plot border and sphere separately (not predicted)
    cells_tensor = torch.tensor(mesh_cells, dtype=torch.long)
    node_type_tensor = torch.tensor(node_type_raw.squeeze(-1), dtype=torch.long)
    dict_traj = {"A": A, "X_seq_norm": X_feat, "mean": 0, "std": 0, "cells": cells_tensor,
                 "node_type": node_type_tensor}

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
    tfrecord_path = dataconfig['tfrecord_path']
    meta_path = dataconfig['meta_path']
    max_trajs = dataconfig['max_trajs']

    if norm_method not in ['centroid', 'standard']:
        raise ValueError(f"norm_method == {norm_method} not supported")

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
        # Stop if we reached max_trajs
        if max_trajs is not None and traj_idx >= max_trajs:
            print("[load_all_trajectories] Reached wanted number of trajectories")
            break

        # DECODE RAW TRAJECTORY
        traj = cast_trajectory_from_record(record, meta)

        dict_traj, X_feat = process_single_trajectory(traj, include_mesh_pos, norm_method, idx)
        list_of_trajs.append(dict_traj)

    # Now that positions are centered per frame/traj, mean is approx 0 for positions.
    # We still compute global mean/std for normalization.

    mean, element_num = compute_global_mean(list_of_trajs)

    if norm_method == "centroid":
        mean, std_dev = compute_centroid_normalization(list_of_trajs, mean, element_num,
                                                       feat_idx, include_mesh_pos)
    else:
        mean, std_dev = compute_standard_normalization(list_of_trajs, mean, element_num,
                                                       feat_idx, include_mesh_pos)

    apply_normalization(list_of_trajs, mean, std_dev)

    print(f"\nLoaded {len(list_of_trajs)} trajectories.")
    return list_of_trajs


if __name__ == "__main__":
    dataconfig_path = os.path.join(os.path.dirname(__file__), "dataconfig.yaml")
    dataconfig = load_config(dataconfig_path)
    list_of_trajs = load_all_trajectories(dataconfig)