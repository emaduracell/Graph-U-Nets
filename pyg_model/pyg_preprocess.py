import os
import json
import torch
import numpy as np
from typing import Optional

from tfrecord.torch.dataset import TFRecordDataset

from pyg_graph_unet_data import (
    _build_description,
    _decode_record,
    _cell_to_edge_index,
    _one_hot_node_type,
    FEATURE_DIM,
    TARGET_DIM,
    NODE_TYPE_START,
    NODE_TYPE_END,
    STATS_FILE,
    INDEX_FILE,
    TRAJ_TEMPLATE,
    PREPROCESSED_DIR,
)


def preprocess(
    data_dir: str,
    split: str = "train",
    num_samples: Optional[int] = None,
    num_steps: Optional[int] = None,
):
    meta_path = os.path.join(data_dir, "meta.json")
    tfrecord_path = os.path.join(data_dir, f"{split}.tfrecord")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at {meta_path}")
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord not found at {tfrecord_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)

    trajectory_length = meta["trajectory_length"]
    if num_steps is None:
        num_steps = trajectory_length
    num_steps = min(num_steps, trajectory_length)

    pre_dir = os.path.join(data_dir, PREPROCESSED_DIR)
    os.makedirs(pre_dir, exist_ok=True)

    description = _build_description(meta)
    raw_ds = TFRecordDataset(
        tfrecord_path,
        index_path=None,
        description=description,
        transform=lambda rec: _decode_record(rec, meta),
    )

    # Accumulate stats
    sum_feat = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    sumsq_feat = torch.zeros(FEATURE_DIM, dtype=torch.float64)
    sum_target = torch.zeros(TARGET_DIM, dtype=torch.float64)
    sumsq_target = torch.zeros(TARGET_DIM, dtype=torch.float64)
    count_feat = 0
    count_target = 0

    index_list = []
    traj_processed = 0

    for traj_idx, rec in enumerate(raw_ds):
        if num_samples is not None and traj_idx >= num_samples:
            break

        world_pos = rec["world_pos"][: num_steps]  # (T, N, 3)
        stress = rec["stress"][: num_steps]        # (T, N, 1)
        node_type = rec["node_type"][0]            # (N, 1)
        cells = rec["cells"][0]                    # (C, 4)

        edge_index = _cell_to_edge_index(cells)
        node_type_oh = _one_hot_node_type(node_type)  # (N, 2)

        vel = np.zeros_like(world_pos)
        vel[1:] = world_pos[1:] - world_pos[:-1]

        time_steps = world_pos.shape[0]
        x_seq = []
        y_seq = []
        for t in range(time_steps - 1):
            x_t = np.concatenate([world_pos[t], node_type_oh, vel[t], stress[t]], axis=-1)
            y_t = np.concatenate([vel[t + 1], stress[t + 1]], axis=-1)
            x_seq.append(x_t)
            y_seq.append(y_t)

            sum_feat += torch.from_numpy(x_t).sum(dim=0)
            sumsq_feat += torch.from_numpy(x_t ** 2).sum(dim=0)
            sum_target += torch.from_numpy(y_t).sum(dim=0)
            sumsq_target += torch.from_numpy(y_t ** 2).sum(dim=0)
            count_feat += x_t.shape[0]
            count_target += y_t.shape[0]

            index_list.append((traj_idx, t))

        # Save raw (unnormalized) sequences for this trajectory
        torch.save(
            {
                "edge_index": edge_index,
                "x_seq": torch.tensor(np.stack(x_seq, axis=0), dtype=torch.float32),
                "y_seq": torch.tensor(np.stack(y_seq, axis=0), dtype=torch.float32),
                "node_type": torch.tensor(node_type.squeeze(-1), dtype=torch.long),
            },
            os.path.join(pre_dir, TRAJ_TEMPLATE.format(traj_idx)),
        )
        traj_processed += 1

    if len(index_list) == 0:
        raise RuntimeError("No samples found during preprocessing.")

    mean_feat = sum_feat / max(count_feat, 1)
    var_feat = sumsq_feat / max(count_feat, 1) - mean_feat ** 2
    std_feat = torch.sqrt(torch.clamp(var_feat, min=1e-8))

    mean_target = sum_target / max(count_target, 1)
    var_target = sumsq_target / max(count_target, 1) - mean_target ** 2
    std_target = torch.sqrt(torch.clamp(var_target, min=1e-8))

    mean_feat[NODE_TYPE_START:NODE_TYPE_END] = 0.0
    std_feat[NODE_TYPE_START:NODE_TYPE_END] = 1.0

    # Second pass: normalize each trajectory file in place
    for traj_idx in range(traj_processed):
        traj_path = os.path.join(pre_dir, TRAJ_TEMPLATE.format(traj_idx))
        if not os.path.exists(traj_path):
            break
        traj = torch.load(traj_path, map_location="cpu")
        traj["x_seq"] = (traj["x_seq"] - mean_feat) / std_feat
        traj["y_seq"] = (traj["y_seq"] - mean_target) / std_target
        torch.save(traj, traj_path)

    torch.save(
        {
            "mean_feat": mean_feat.float(),
            "std_feat": std_feat.float(),
            "mean_target": mean_target.float(),
            "std_target": std_target.float(),
        },
        os.path.join(pre_dir, STATS_FILE),
    )

    torch.save(
        {
            "index": index_list,
        },
        os.path.join(pre_dir, INDEX_FILE),
    )

    print(f"Preprocessing complete. Saved to {pre_dir}")


if __name__ == "__main__":
    preprocess(
        data_dir=os.path.join(os.path.dirname(__file__), "data"),
        split="train",
        num_samples=None,
        num_steps=None,
    )

