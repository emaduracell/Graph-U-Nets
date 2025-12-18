from pathlib import Path

import numpy as np
import tensorflow as tf
import torch
from tqdm import tqdm

TFRECORD_LOCATION = Path("datasets/deforming_plate/train.tfrecord")

def deforming_plate_tfrecord_to_torch_dataset(tfrecord_path):
    raw_dataset = tf.data.TFRecordDataset([tfrecord_path])
    for sample_idx, raw in enumerate(raw_dataset):
        ex = tf.train.Example()
        ex.ParseFromString(raw.numpy())
        f = ex.features.feature

        def arr(name, dtype=np.float32):
            return np.frombuffer(f[name].bytes_list.value[0], dtype=dtype)

        mesh_pos = arr("mesh_pos").reshape(-1, 3)
        world_pos = arr("world_pos").reshape(-1, 3)
        T = world_pos.size // (mesh_pos.shape[0] * 3)
        world_pos = world_pos.reshape(T, mesh_pos.shape[0], 3)
        raw_cells = arr("cells", np.int32)
        try:
            cells = raw_cells.reshape(-1, 4)  # tetrahedral cells
        except ValueError:
            print(f"Skipping sample {sample_idx}: Invalid cells size {raw_cells.size}")
            cells = None
        node_type = arr("node_type", np.int32)
        stress = arr("stress")
        if stress.size == T * mesh_pos.shape[0]:
            stress = stress.reshape(T, mesh_pos.shape[0])
        elif stress.size == T * mesh_pos.shape[0] * 6:
            stress = stress.reshape(T, mesh_pos.shape[0], 6)

        yield {
            "sample_idx": sample_idx,
            "mesh_pos": torch.from_numpy(np.copy(mesh_pos)),
            "world_pos": torch.from_numpy(np.copy(world_pos)),
            "cells": torch.from_numpy(np.copy(cells)),
            "node_type": torch.from_numpy(np.copy(node_type)),
            "stress": torch.from_numpy(np.copy(stress)),
        }


dataset_dir = Path("datasets/deforming_plate")
dataset_dir.mkdir(parents=True, exist_ok=True)
partition = "train"
print()
print(f"Processing partition: {partition}")
dataset = deforming_plate_tfrecord_to_torch_dataset(dataset_dir / f"{partition}.tfrecord")

# Create a list to store geometric data objects
geometric_data_objects = []

for entry in tqdm(dataset, desc="Processing dataset"):
    # Create a geometric data object (example structure, adjust as needed)
    geometric_data = {"sample_idx": entry["sample_idx"],
        "mesh_pos": entry["mesh_pos"],
        "world_pos": entry["world_pos"],
        "cells": entry["cells"],
        "node_type": entry["node_type"],
        "stress": entry["stress"]
    }
    geometric_data_objects.append(geometric_data)

# Save the list of geometric data objects to a .pth file
torch.save(geometric_data_objects, dataset_dir / f"{partition}.pth")

print(f"Saved dataset to {dataset_dir / f'{partition}.pth'}")
