import json
import numpy as np
import torch
from tfrecord.reader import tfrecord_loader

# 1) DECODIFICA RECORD RAW (NO TENSORFLOW)

def decode_trajectory_from_record(record, meta):
    """
    Decodifica un record da tfrecord_loader gestendo:
    - bytes
    - np.ndarray numerici
    - np.ndarray(object) contenenti bytes

    Usa np.frombuffer + reshape usando le shape nel meta.json.
    """

    def normalize_to_bytes(value):
        """
        Estrae sempre dei bytes, ma SOLO se value NON è già un ndarray numerico.
        """
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)

        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return bytes(value.flat[0])
            # ndarray numerico → NON gestito qui
            raise TypeError("normalize_to_bytes chiamato su ndarray numerico")

        raise TypeError(f"Tipo inatteso: {type(value)}")

    def decode_raw_array(value, dtype, shape_spec):
        """
        value può essere:
        - np.ndarray numerico → restituito (cast + reshape)
        - bytes / np.ndarray(object) → np.frombuffer + reshape
        """

        # Caso 1: già ndarray numerico
        if isinstance(value, np.ndarray) and value.dtype != object:
            arr = value.astype(dtype)
            # costruiamo la shape corretta
            tgt = list(shape_spec)
            if -1 in tgt:
                known = 1
                for d in tgt:
                    if d != -1:
                        known *= d
                missing = arr.size // known
                tgt[tgt.index(-1)] = missing
            if tuple(arr.shape) != tuple(tgt):
                arr = arr.reshape(tgt)
            return arr

        # Caso 2: è bytes o array di object → frombuffer
        raw = normalize_to_bytes(value)
        arr = np.frombuffer(raw, dtype=dtype)

        tgt = list(shape_spec)
        if -1 in tgt:
            known = 1
            for d in tgt:
                if d != -1:
                    known *= d
            missing = arr.size // known
            tgt[tgt.index(-1)] = missing

        return arr.reshape(tgt)

    shapes = meta["features"]

    world_pos = decode_raw_array(record["world_pos"], np.float32,
                                 shapes["world_pos"]["shape"])
    stress = decode_raw_array(record["stress"], np.float32,
                              shapes["stress"]["shape"])
    node_type = decode_raw_array(record["node_type"], np.int32,
                                 shapes["node_type"]["shape"])
    mesh_pos = decode_raw_array(record["mesh_pos"], np.float32,
                                shapes["mesh_pos"]["shape"])
    cells = decode_raw_array(record["cells"], np.int32,
                             shapes["cells"]["shape"])

    # Rimuovi la prima dimensione =1 per statiche
    if node_type.shape[0] == 1:
        node_type = node_type[0]
    if mesh_pos.shape[0] == 1:
        mesh_pos = mesh_pos[0]
    if cells.shape[0] == 1:
        cells = cells[0]

    return {
        "world_pos": world_pos.astype(np.float32),  # (T, N, 3)
        "stress": stress.astype(np.float32),        # (T, N, 1)
        "node_type": node_type.astype(np.int32),    # (N, 1)
        "mesh_pos": mesh_pos.astype(np.float32),    # (N, 3)
        "cells": cells.astype(np.int32),            # (C, 4)
    }


def load_raw_trajectory_from_tfrecord(tfrecord_path, meta, traj_index):
    loader = tfrecord_loader(tfrecord_path, index_path=None)

    for i, record in enumerate(loader):
        if i == traj_index:
            return decode_trajectory_from_record(record, meta)

    raise IndexError(f"trajectory index {traj_index} out of range")



def build_edges_from_cells(cells, num_nodes):
    edge_set = set()

    for c in cells:
        i0, i1, i2, i3 = map(int, c.tolist())
        quad = [i0, i1, i2, i3]

        for u, v in zip(quad, quad[1:] + quad[:1]):
            if u != v:
                edge_set.add((u, v))
                edge_set.add((v, u))

    edge_list = sorted(edge_set)
    return torch.tensor(edge_list, dtype=torch.long)

# 3) ASSEMBLAGGIO INPUT EGNN

def trajectory_to_egnn_inputs(traj):
    world_pos = traj["world_pos"]   # (T,N,3)
    stress = traj["stress"]         # (T,N,1)
    node_type = traj["node_type"]   # (N,1)
    cells = traj["cells"]           # (C,4)

    T, N, _ = world_pos.shape

    coors_seq = torch.tensor(world_pos, dtype=torch.float32)

    vel = np.zeros((T, N, 3), dtype=np.float32)
    acc = np.zeros((T, N, 3), dtype=np.float32)

    if T > 1:
        vel[1] = world_pos[1] - world_pos[0]

    for t in range(2, T):
        vel[t] = world_pos[t] - world_pos[t - 1]
        acc[t] = world_pos[t] - 2 * world_pos[t - 1] + world_pos[t - 2]

    node_type_f = node_type.astype(np.float32)

    feats = []
    for t in range(T):
        feats_t = np.concatenate(
            [
                node_type_f,    # (N,1)
                vel[t],         # (N,3)
                #acc[t],         # (N,3)
                stress[t],      # (N,1)
            ],
            axis=-1
        )
        feats.append(feats_t)

    feats_seq = torch.tensor(np.stack(feats, axis=0), dtype=torch.float32)

    edge_index = build_edges_from_cells(cells, num_nodes=N)

    return coors_seq, feats_seq, edge_index

# 4) WRAPPER FINALE
def data_loader_egnn(tfrecord_path, meta_path, traj_index):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    traj_dict = load_raw_trajectory_from_tfrecord(tfrecord_path, meta, traj_index)
    return trajectory_to_egnn_inputs(traj_dict)

# 5) TEST
if __name__ == "__main__":
    tfrecord_path = "data/deforming_plate/train.tfrecord"
    meta_path = "data/deforming_plate/meta.json"

    coors, feats, edge_index = data_loader_egnn(
        tfrecord_path, meta_path, traj_index=0
    )

    print("coors_seq:", coors.shape)
    print("feats_seq:", feats.shape)
    print("edge_index:", edge_index.shape)