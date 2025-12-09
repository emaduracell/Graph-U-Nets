import json
import torch
import numpy as np
from tfrecord.reader import tfrecord_loader

NORMAL_NODE = [0, 0]  # value 0 (NORMAL)
SPHERE_NODE = [1, 0]  # value 1 (SPHERE)
BOUNDARY_NODE = [0, 1]  # value 3 (BOUNDARY)

def _cast_to_bytes(value):
    """
    Decodes a value of type bytearray to bytes, with the goal of making it immutable. If value is of type ndarray,
    and its elements are objects of python (not numpy objects), it flattens it to an iterator (the assumption is that
    we don't lose info because the shape is (,1) or (1,1)), and gets the first element.

    :param value: np.ndarray | (bytes, bytearray) | Any
         input value
    :return: an array of bytes of the value
    """
    if isinstance(value, (bytes, bytearray)):
        return bytes(value)
    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.size != 1:
                raise ValueError(f"Expected a single bytes object, got {value.size}")
            return bytes(value.flat[0])
        raise TypeError("normalize_to_bytes called on numeric ndarray")
    else:
        raise TypeError(f"Unexpected type: {type(value)}")


def _reshape_with_inferred_dim(arr, shape_spec):
    """
    Reshape arr according to shape_spec, where at most one entry may be -1 (to be inferred from arr.size).

    :param arr: np.ndarray
    :param shape_spec

    :return arr.reshape(shape)
        reshaped np.ndarray
    """
    shape = list(shape_spec)
    # sanity check
    if shape.count(-1) > 1:
        raise ValueError("At most one -1 is allowed in shape_spec")
    # infer missing shape
    if -1 in shape:
        known = int(np.prod([d for d in shape if d != -1])) or 1
        if arr.size % known != 0:
            raise ValueError(f"Cannot infer missing dimension: {arr.size} elements "
                             f"is not divisible by known product {known}")
        inferred = arr.size // known
        shape[shape.index(-1)] = inferred
    # sanity check
    if np.prod(shape) != arr.size:
        raise ValueError(f"Cannot reshape array of size {arr.size} into shape {tuple(shape)}")
    # Avoid unnecessary reshape if shape already matches
    if tuple(arr.shape) == tuple(shape):
        return arr

    return arr.reshape(shape)


def cast_raw_array(value, dtype, shape_spec):
    """
    Cast an array to a np.ndarray.
    It splits in two cases, one if (value is np.ndarray AND value.dtype != object), the other one is the else case.
    For both do: 1. Cast to dtype 2. Reshape

    :param value:
        input value that I want to decode
    :param dtype:
        dtype I want to convert value in.
    :param shape_spec:
        shape I want to convert value in.

    :return converted_arr: np.ndarray
        array converted to desired
    """
    # Cast to dtype
    if isinstance(value, np.ndarray) and value.dtype != object:
        # If it's already a numpy array and has numpy values/elements
        converted_arr = value.astype(dtype)
    else:
        # If it's not a numpy array or it has python values/elements
        raw_bytes = _cast_to_bytes(value)
        # Re-read raw bytes as for the desired dtype
        converted_arr = np.frombuffer(raw_bytes, dtype=dtype)
    # Reshape
    converted_arr = _reshape_with_inferred_dim(converted_arr, shape_spec)
    return converted_arr


def cast_trajectory_from_record(record, meta):
    """
    Casts the TFRecord into numpy arrays.
    The feature meta["features"] is a dict that contains info about the features and their shape.

    :param record:
        A TFRecord object
    :param meta:
        json file

    :return trajectory_dict: Dict
        A dict containing all trajectory features
    """

    shapes = meta["features"]
    world_pos = cast_raw_array(record["world_pos"], np.float32, shapes["world_pos"]["shape"])
    stress = cast_raw_array(record["stress"], np.float32, shapes["stress"]["shape"])
    node_type = cast_raw_array(record["node_type"], np.int32, shapes["node_type"]["shape"])
    mesh_pos = cast_raw_array(record["mesh_pos"], np.float32, shapes["mesh_pos"]["shape"])
    mesh_cells = cast_raw_array(record["cells"], np.int32, shapes["cells"]["shape"])

    # remove extra dims
    if node_type.shape[0] == 1:
        node_type = node_type[0]
    if mesh_pos.shape[0] == 1:
        mesh_pos = mesh_pos[0]
    if mesh_cells.shape[0] == 1:
        mesh_cells = mesh_cells[0]
    trajectory_dict = {
        "world_pos": world_pos.astype(np.float32),
        "stress": stress.astype(np.float32),
        "node_type": node_type.astype(np.int32),
        "mesh_pos": mesh_pos.astype(np.float32),
        "cells": mesh_cells.astype(np.int32),
    }
    return trajectory_dict


def build_edges_from_cells(mesh_cells):
    """
    Receives the set of all mesh cell, each mesh cell is made by 4 points
    0. Declares an empty set to not count edge duplicates
    1. for all cells: convert to standard python list with ints and unpack them in 4 variables
    2. It sorts edge list and then converts to a torch tensor

    :param mesh_cells: (some kind of collection)
        some kind of collection of mesh cells

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


def load_all_trajectories(tfrecord_path, meta_path, max_trajs):
    """
    Load up to `max_trajs` trajectories from TFRecord.

    :param tfrecord_path: str
        path of the tfrecord files
    :param meta_path: str
        path of the meta.json file
    :param max_trajs: int
        maximum number of trajectories to load

    :return list_of_trajs: List
        list of dicts where each dict contains:
          - "A"          : [N,N] adjacency matrix (torch.float32)
          - "X_seq_norm" : [T,N,F] normalized features (torch.float32)
          - "mean"       : [1,1,F] mean for denorm
          - "std"        : [1,1,F] std for denorm
          - "cells"      : [C,4] connectivity (torch.long)
    """

    # Load meta.json for decoding
    with open(meta_path, "r") as f:
        meta = json.load(f)
    # TFRecord loader
    loader = tfrecord_loader(tfrecord_path, index_path=None)
    list_of_trajs = []
    idx = 0  # debug idx
    sum_elements = 0
    element_num = 0
    # Iterate through trajectories
    for traj_idx, record in enumerate(loader):
        # Stop if we reached max_trajs
        if max_trajs is not None and traj_idx >= max_trajs:
            print("[load_all_trajectories] Reached wanted number of trajectories")
            break

        # DECODE RAW TRAJECTORY 
        traj = cast_trajectory_from_record(record, meta)
        world_pos = traj["world_pos"]  # (T,N,3)
        stress = traj["stress"]  # (T,N,1)
        node_type = traj["node_type"]  # (N,1)
        mesh_cells = traj["cells"]  # (C,4)
        if idx == 0 or idx == 1 or idx == 2:
            print(f"traj: \n \t type(traj) = {type(traj)}, len={len(traj)}")
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

        time_step_dim, number_of_nodes, _ = world_pos.shape

        # Build velocity
        vel = np.zeros((time_step_dim, number_of_nodes, 3), dtype=np.float32)
        for t in range(1, time_step_dim):
            vel[t] = world_pos[t] - world_pos[t - 1]

        lookup = np.array([
            NORMAL_NODE,  # value 0 (NORMAL)
            SPHERE_NODE,  # value 1 (SPHERE)
            [0, 0],  # value 2 (not used)
            BOUNDARY_NODE,  # value 3 (BOUNDARY)
        ])
        # Keep a copy of the raw scalar node types for plotting later
        node_type_raw = node_type.copy()  # shape (N, 1)
        node_type_idx = node_type_raw.squeeze(-1)  # (N,)
        node_type_onehot = lookup[node_type_idx]  # (N, 2)

        if idx == 1 or idx == 2:
            print(
                f"node_type: \n \t type(node_type) = {type(node_type)} \n \t type(node_type[0])={type(node_type[0])}, "
                f"\n \t type(node_type[0][0])={type(node_type[0][0])}) \n \t len(node_type)={len(node_type)} "
                f"\n \t len(node_type[0])={len(node_type[0])}")

        # Build feature sequence Feature layout: [pos_x, pos_y, pos_z, node_type, vel_x, vel_y, vel_z, stress]
        feats_list = []
        node_type_floatcast = node_type_onehot.astype(np.float32)

        for t in range(time_step_dim):
            feats_t = np.concatenate([world_pos[t], node_type_floatcast, vel[t], stress[t]], axis=-1)
            feats_list.append(feats_t)
        X_seq = torch.tensor(np.stack(feats_list, axis=0), dtype=torch.float32)
        # print(f"X_seq.shape={X_seq.shape}")

        sum_elements = sum_elements + X_seq.sum(dim=(0, 1))
        element_num = element_num + X_seq.shape[0] * X_seq.shape[1]

        # Build adjacency matrix from set + add self loops + row normalize
        edge_index = build_edges_from_cells(mesh_cells)
        A = torch.zeros((number_of_nodes, number_of_nodes), dtype=torch.float32)
        for e in edge_index:
            A[e[0], e[1]] = 1.0
        A = A + torch.eye(number_of_nodes)
        # adj_sanity(A, mesh_cells)
        A = A / A.sum(dim=1, keepdim=True)
        # sanity checks before normalization
        row_sums = A.sum(dim=1)
        if not torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-5):
            print(f"[load_all_trajectories] BAD A just built for traj {traj_idx}")
            print("  min row sum:", row_sums.min().item())
            print("  max row sum:", row_sums.max().item())
            # Optionally inspect mesh_cells as well
            print("  mesh_cells shape:", mesh_cells.shape)
            print("  max index in mesh_cells:", mesh_cells.max())
            print("  number_of_nodes:", number_of_nodes)
            raise RuntimeError("Adjacency normalization failed immediately after construction.")
        # ensure cells and node_type are tensors, passing them to plot border and sphere separately (not predicted)
        cells_tensor = torch.tensor(mesh_cells, dtype=torch.long)
        node_type_tensor = torch.tensor(node_type_raw.squeeze(-1), dtype=torch.long)
        list_of_trajs.append({
            "A": A,
            "X_seq_norm": X_seq,
            "mean": 0,
            "std": 0,
            "cells": cells_tensor,  # passing them to plot border and sphere separately (not predicted)
            "node_type": node_type_tensor  # passing them to plot border and sphere separately (not predicted)
        })
        # print(f"Loaded trajectory {traj_idx}")

    mean = sum_elements / element_num
    std_acc = torch.zeros_like(mean)
    for traj in list_of_trajs:
        X = traj['X_seq_norm']
        std_acc += ((X - mean.view(1, 1, -1)) ** 2).sum(dim=(0, 1))

    std_dev = torch.sqrt(std_acc / (element_num - 1))

    max_std_pos = std_dev[0:3].max()
    std_dev[0:3] = max_std_pos
    max_std_vel = std_dev[5:8].max()
    std_dev[5:8] = max_std_vel

    NODE_TYPE_START = 3
    NODE_TYPE_END = 5
    mean[NODE_TYPE_START:NODE_TYPE_END] = 0.0
    std_dev[NODE_TYPE_START:NODE_TYPE_END] = 1.0

    mean_b = mean.view(1, 1, -1)
    std_b = std_dev.view(1, 1, -1)

    for traj in list_of_trajs:
        traj['mean'] = mean_b
        traj['std'] = std_b
        X = traj['X_seq_norm']
        X_seq_norm = (X - mean_b) / std_b
        traj['X_seq_norm'] = X_seq_norm

    print(f"\nLoaded {len(list_of_trajs)} trajectories. "
          f"")
    return list_of_trajs


if __name__ == "__main__":
    TFRECORD_PATH = "data/train.tfrecord"
    META_PATH = "data/meta.json"
    NUM_TRAIN_TRAJS = 1500  # Load only the first K trajectories
    list_of_trajs = load_all_trajectories(TFRECORD_PATH, META_PATH, NUM_TRAIN_TRAJS)
