import json
import torch
import numpy as np
from tfrecord.reader import tfrecord_loader


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
    Casts the TFRecord into numpy arrays. TODO FLOAT 32???
    The feature meta["features"] is a dict that contains info about the features and their shape.

    :param record: TODO ??
        A TFRecord object
    :param meta: TODO ??
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
    1. for all cells: 1.1 convert to standard python list with ints and unpack them in 4 variables 1.2
    2. It sorts edge list and then converts to a torch tensor

    :param mesh_cells: (some kind of collection)
        some kind of collection of mesh cells

    :return edge_list: torch.tensor
        torch tensor with shape (#edges, 2) since each edge i-->j is (i, j)
    """
    edge_set = set()
    for c in mesh_cells:
        # unpack and repack
        i0, i1, i2, i3 = map(int, c.tolist())
        quad = [i0, i1, i2, i3]
        # for all
        for u, v in zip(quad, quad[1:] + quad[:1]):
            if u != v:
                edge_set.add((u, v))
                edge_set.add((v, u))
            # TODO NO SELF LOOPS?

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
    idx = 0 # debug idx

    # Iterate through trajectories
    for traj_idx, record in enumerate(loader):
        # Stop if we reached max_trajs
        if max_trajs is not None and traj_idx >= max_trajs:
            print("[load_all_trajectories] Reached maximum trajectories")
            break

        # DECODE RAW TRAJECTORY 
        traj = cast_trajectory_from_record(record, meta)
        world_pos = traj["world_pos"]  # (T,N,3)
        stress = traj["stress"]  # (T,N,1)
        node_type = traj["node_type"]  # (N,1)
        mesh_cells = traj["cells"]  # (C,4)
        if idx == 0:
            print(f"type(traj) = {type(traj)}")
            print(f"type(world_pos) = {type(world_pos)}, type(world_pos[0])={type(world_pos[0])}, "
                  f"type(world_pos[0][0])={type(world_pos[0][0])})")
            print(f"type(stress) = {type(stress)}, type(stress[0])={type(stress[0])}, "
                  f"type(stress[0][0])={type(stress[0][0])})")
            print(f"type(node_type) = {type(node_type)}, type(node_type[0])={type(node_type[0])}, "
                  f"type(node_type[0][0])={type(node_type[0][0])})")
            print(f"type(mesh_cells) = {type(mesh_cells)}, type(mesh_cells[0])={type(mesh_cells[0])}, "
                  f"type(mesh_cells[0][0])={type(mesh_cells[0][0])})")
            idx += 1

        T, N, _ = world_pos.shape

        # BUILD VELOCITY 
        vel = np.zeros((T, N, 3), dtype=np.float32)
        for t in range(1, T):
            vel[t] = world_pos[t] - world_pos[t - 1]

        # BUILD FEATURE SEQUENCE Feature layout: [pos_x, pos_y, pos_z, node_type, vel_x, vel_y, vel_z, stress]
        feats_list = []
        node_type_f = node_type.astype(np.float32)

        for t in range(T):
            feats_t = np.concatenate([
                    world_pos[t],
                    node_type_f,
                    vel[t],
                    stress[t],
                ],
                axis=-1
            )
            feats_list.append(feats_t)

        X_seq = torch.tensor(np.stack(feats_list, axis=0), dtype=torch.float32)

        # NORMALIZATION PER TRAJECTORY TODO FIX IT ACROSS DATASET
        mean = X_seq.mean(dim=(0, 1), keepdim=True)
        std = X_seq.std(dim=(0, 1), keepdim=True)
        X_seq_norm = (X_seq - mean) / std

        # Adjacency matrix
        edge_index = build_edges_from_cells(mesh_cells)

        A = torch.zeros((N, N), dtype=torch.float32)
        for e in edge_index:
            A[e[0], e[1]] = 1.0

        # add self-loops
        A = A + torch.eye(N)
        # row-normalize
        A = A / A.sum(dim=1, keepdim=True)
        # ensure cells and node_type are tensors
        cells_t = torch.tensor(mesh_cells, dtype=torch.long)
        node_type_t = torch.tensor(node_type.squeeze(-1), dtype=torch.long)

        # Store trajectory in the trajectory list
        list_of_trajs.append({
            "A": A,
            "X_seq_norm": X_seq_norm,
            "mean": mean,
            "std": std,
            "cells": cells_t,
            "node_type": node_type_t
        })

        print(f"Loaded trajectory {traj_idx}, shape = {X_seq_norm.shape}")

    print(f"\nLoaded {len(list_of_trajs)} trajectories.")
    return list_of_trajs

if __name__ == "__main__":
    TFRECORD_PATH = "data/train.tfrecord"
    META_PATH = "data/meta.json"
    NUM_TRAIN_TRAJS = 1500  # Load only the first K trajectories
    list_of_trajs = load_all_trajectories(TFRECORD_PATH, META_PATH, NUM_TRAIN_TRAJS)