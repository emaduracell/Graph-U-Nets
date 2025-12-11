import os
import json
import yaml
import torch
import numpy as np
from typing import Optional
from tfrecord.torch.dataset import TFRecordDataset

from pyg_data import (
    _build_description, _decode_record, _cell_to_edge_index, _one_hot_node_type,
    FEATURE_DIM, TARGET_DIM, NODE_TYPE_START, NODE_TYPE_END,
    STATS_FILE, INDEX_FILE, TRAJ_TEMPLATE
)

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def preprocess(
    data_dir: str,
    output_dir: str,
    split: str = "train",
    norm_method: str = "centroid",
    include_mesh_pos: bool = True,
    num_samples: Optional[int] = None,
    num_steps: Optional[int] = None,
):
    print(f"\n--- Starting Preprocessing ---")
    print(f"Data Source:  {data_dir}")
    print(f"Output Dir:   {output_dir}")
    print(f"Config:       Method='{norm_method}', Mesh_Pos={include_mesh_pos}")
    
    meta_path = os.path.join(data_dir, "meta.json")
    tfrecord_path = os.path.join(data_dir, f"{split}.tfrecord")
    
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"meta.json not found at {os.path.abspath(meta_path)}")
    if not os.path.exists(tfrecord_path):
        raise FileNotFoundError(f"TFRecord not found at {os.path.abspath(tfrecord_path)}")

    with open(meta_path, "r") as f:
        meta = json.load(f)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    trajectory_length = meta["trajectory_length"]
    if num_steps is None: num_steps = trajectory_length
    else: num_steps = min(num_steps, trajectory_length)

    description = _build_description(meta)
    raw_ds = TFRecordDataset(tfrecord_path, index_path=None, description=description,
                             transform=lambda rec: _decode_record(rec, meta))

    # Determine Feature Size for stats
    # 12 if mesh_pos included, 9 if not.
    current_dim = 12 if include_mesh_pos else 9
    
    sum_feat = torch.zeros(current_dim, dtype=torch.float64)
    sumsq_feat = torch.zeros(current_dim, dtype=torch.float64)
    sum_target = torch.zeros(TARGET_DIM, dtype=torch.float64)
    sumsq_target = torch.zeros(TARGET_DIM, dtype=torch.float64)
    count_feat = 0
    count_target = 0
    index_list = []
    traj_processed = 0

    for traj_idx, rec in enumerate(raw_ds):
        if num_samples is not None and traj_idx >= num_samples: break

        # 1. Extract Raw Data
        world_pos = rec["world_pos"][: num_steps]
        stress = rec["stress"][: num_steps]
        node_type = rec["node_type"][0]
        cells = rec["cells"][0]
        
        # Handle Mesh Pos
        mesh_pos = None
        if include_mesh_pos:
            if "mesh_pos" in rec:
                mesh_pos_static = rec["mesh_pos"][0]
                mesh_pos = np.tile(mesh_pos_static, (num_steps, 1, 1))
            else:
                # Fallback to initial world pos
                mesh_pos_static = world_pos[0]
                mesh_pos = np.tile(mesh_pos_static, (num_steps, 1, 1))

        vel = np.zeros_like(world_pos)
        vel[1:] = world_pos[1:] - world_pos[:-1]

        edge_index = _cell_to_edge_index(cells)
        node_type_oh = _one_hot_node_type(node_type)

        x_seq = []
        y_seq = []

        # 2. Process Frames (Applying Centroid Logic)
        for t in range(num_steps - 1):
            
            curr_world = world_pos[t]
            curr_vel = vel[t]
            curr_mesh = mesh_pos[t] if include_mesh_pos else None
            
            if norm_method == "centroid":
                # A. Center World Pos
                centroid_world = curr_world.mean(axis=0)
                curr_world = curr_world - centroid_world
                
                # B. Center Velocity
                centroid_vel = curr_vel.mean(axis=0)
                curr_vel = curr_vel - centroid_vel
                
                # C. Center Mesh Pos (if exists)
                if include_mesh_pos:
                    centroid_mesh = curr_mesh.mean(axis=0)
                    curr_mesh = curr_mesh - centroid_mesh
            
            # Construct Feature Vector
            features = []
            if include_mesh_pos:
                features.append(curr_mesh)
            
            features.append(curr_world)
            features.append(node_type_oh)
            features.append(curr_vel)
            features.append(stress[t])
            
            x_t = np.concatenate(features, axis=-1)

            # Target Construction
            y_t = np.concatenate([vel[t + 1], stress[t + 1]], axis=-1)

            x_seq.append(x_t)
            y_seq.append(y_t)
            
            # Update Stats
            x_t_torch = torch.from_numpy(x_t)
            y_t_torch = torch.from_numpy(y_t)
            
            sum_feat += x_t_torch.sum(dim=0)
            sumsq_feat += (x_t_torch ** 2).sum(dim=0)
            sum_target += y_t_torch.sum(dim=0)
            sumsq_target += (y_t_torch ** 2).sum(dim=0)
            
            count_feat += x_t.shape[0]
            count_target += y_t.shape[0]
            index_list.append((traj_idx, t))

        # Save Raw Sequence
        torch.save({
            "edge_index": edge_index,
            "x_seq": torch.tensor(np.stack(x_seq, axis=0), dtype=torch.float32),
            "y_seq": torch.tensor(np.stack(y_seq, axis=0), dtype=torch.float32),
            "node_type": torch.tensor(node_type.squeeze(-1), dtype=torch.long),
        }, os.path.join(output_dir, TRAJ_TEMPLATE.format(traj_idx)))
        
        traj_processed += 1

    # 3. Compute Stats
    mean_feat = sum_feat / max(count_feat, 1)
    var_feat = sumsq_feat / max(count_feat, 1) - mean_feat ** 2
    mean_target = sum_target / max(count_target, 1)
    var_target = sumsq_target / max(count_target, 1) - mean_target ** 2

    # Define Slices
    if include_mesh_pos:
        S_MESH = slice(0, 3)
        S_WORLD = slice(3, 6)
        S_TYPE = slice(6, 8)
        S_VEL = slice(8, 11)
    else:
        S_WORLD = slice(0, 3)
        S_TYPE = slice(3, 5)
        S_VEL = slice(5, 8)

    # Apply Isotropic Scaling logic
    if norm_method == "centroid":
        mean_feat[S_WORLD] = 0.0
        mean_feat[S_VEL] = 0.0
        if include_mesh_pos:
            mean_feat[S_MESH] = 0.0
        
        def make_isotropic(variance_vec, s):
            return torch.sqrt(variance_vec[s].mean())

        std_feat = torch.sqrt(var_feat)
        std_feat[S_WORLD] = make_isotropic(var_feat, S_WORLD)
        std_feat[S_VEL] = make_isotropic(var_feat, S_VEL)
        if include_mesh_pos:
            std_feat[S_MESH] = make_isotropic(var_feat, S_MESH)
            
        target_vel_std = make_isotropic(var_target, slice(0, 3))
        std_target = torch.sqrt(var_target)
        std_target[slice(0, 3)] = target_vel_std
    else:
        std_feat = torch.sqrt(torch.clamp(var_feat, min=1e-8))
        std_target = torch.sqrt(torch.clamp(var_target, min=1e-8))

    # Keep Node Types 0/1
    mean_feat[S_TYPE] = 0.0
    std_feat[S_TYPE] = 1.0

    # 4. Normalize & Rewrite
    print(f"Stats computed. Normalizing {traj_processed} files...")
    for traj_idx in range(traj_processed):
        path = os.path.join(output_dir, TRAJ_TEMPLATE.format(traj_idx))
        if not os.path.exists(path): break
        traj = torch.load(path)
        traj["x_seq"] = (traj["x_seq"] - mean_feat.float()) / std_feat.float()
        traj["y_seq"] = (traj["y_seq"] - mean_target.float()) / std_target.float()
        torch.save(traj, path)

    torch.save({
        "mean_feat": mean_feat.float(),
        "std_feat": std_feat.float(),
        "mean_target": mean_target.float(),
        "std_target": std_target.float(),
    }, os.path.join(output_dir, STATS_FILE))
    
    torch.save({"index": index_list}, os.path.join(output_dir, INDEX_FILE))
    print(f"Preprocessing Complete. Saved to {output_dir}")

if __name__ == "__main__":
    # 1. Load Configuration
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    data_cfg = config['data']
    
    # 2. Resolve Paths (Relative to this script)
    # The config says "data_dir: ../pytorch_model/raw_data"
    # We join it with current file dir to get absolute path
    base_dir = os.path.dirname(__file__)
    
    raw_data_path = os.path.join(base_dir, data_cfg['data_dir'])
    processed_data_path = os.path.join(base_dir, data_cfg['preprocessed_dir'])
    
    # 3. Run Preprocess using Config Values
    preprocess(
        data_dir=raw_data_path,
        output_dir=processed_data_path,
        split=data_cfg.get('split', 'train'),
        norm_method=data_cfg.get('normalization_method', 'centroid'),
        include_mesh_pos=data_cfg.get('include_mesh_pos', True)
    )