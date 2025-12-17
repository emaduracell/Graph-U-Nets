import os
import json
import torch
import numpy as np
import shutil
import sys
from tfrecord.reader import tfrecord_loader
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.nn import radius_graph, knn_graph
from torch_geometric.utils import to_undirected, coalesce
from helpers.helpers import get_feature_indices, load_config
from data.decode_tfrecord_utils import cast_trajectory_from_record

# Constants
NORMAL_NODE_OH = [0, 0]
SPHERE_NODE_OH = [1, 0]
BOUNDARY_NODE_OH = [0, 1]
NORMAL_NODE = 0
SPHERE_NODE = 1
BOUNDARY_NODE = 3


def build_mesh_edge_index(mesh_cells):
    """Vectorized conversion of mesh cells (quads) to COO edge_index."""
    cells = torch.tensor(mesh_cells, dtype=torch.long)
    s = torch.cat([cells[:, 0], cells[:, 0], cells[:, 0], cells[:, 1], cells[:, 1], cells[:, 2]])
    t = torch.cat([cells[:, 1], cells[:, 2], cells[:, 3], cells[:, 2], cells[:, 3], cells[:, 3]])
    edge_index = torch.stack([s, t], dim=0)
    edge_index = to_undirected(edge_index)
    return edge_index


def build_velocity(world_pos, mode):
    if mode not in ["normal", "actuator"]:
        raise ValueError(f"Unknown mode = {mode}")
    vel = np.zeros_like(world_pos)
    if mode == "normal":
        vel[1:] = world_pos[1:] - world_pos[:-1]
    elif mode == "actuator":
        vel[1:-1] = world_pos[2:] - world_pos[1:-1]
    return vel


class TrajectoryDataset(InMemoryDataset):
    def __init__(self, root, dataconfig, transform, pre_transform):
        self.dataconfig = dataconfig
        # PyG will check root/processed/processed_file_names
        # If found, it skips process(). If not, it runs process().
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # We don't use raw_dir logic for TFRecords, return empty to skip raw check
        return []

    @property
    def processed_file_names(self):
        return ['pyg_processed_data.pt']

    def process(self):
        print(f"Starting processing... Saving to {self.processed_paths[0]}")

        # Config loading
        tfrecord_path = self.dataconfig['tfrecord_path']
        meta_path = self.dataconfig['meta_path']
        max_trajs = self.dataconfig.get('max_trajs')
        include_mesh_pos = self.dataconfig.get('include_mesh_pos')
        norm_method = self.dataconfig.get('normalization_method')

        # World Edge Config
        add_world_edges = self.dataconfig.get('add_world_edges')
        radius = self.dataconfig.get('radius_world_edge')
        k_neighb = self.dataconfig.get('k_neighb')

        with open(meta_path, "r") as f:
            meta = json.load(f)

        loader = tfrecord_loader(tfrecord_path, index_path=None)
        data_list = []

        # Create data objects
        print("Parsing TFRecords...")
        for i, record in enumerate(loader):
            if max_trajs is not None and i >= max_trajs:
                break

            traj = cast_trajectory_from_record(record, meta)

            world_pos = traj["world_pos"]
            node_type = traj["node_type"]
            stress = traj["stress"]
            mesh_cells = traj["cells"]
            mesh_pos = traj["mesh_pos"] if include_mesh_pos else None

            T, N, _ = world_pos.shape

            # Velocities
            vel_normal = build_velocity(world_pos, "normal")
            vel_actuator = build_velocity(world_pos, "actuator")
            actuator_mask = (node_type == SPHERE_NODE).reshape(-1)
            vel_normal[:, actuator_mask, :] = vel_actuator[:, actuator_mask, :]

            # One Hot
            lookup = np.array([NORMAL_NODE_OH, SPHERE_NODE_OH, [0, 0], BOUNDARY_NODE_OH])
            node_type_idx = node_type.squeeze(-1)
            node_type_oh = lookup[node_type_idx].astype(np.float32)

            # Build Features
            feats_list = []
            for t in range(T):
                components = []
                if include_mesh_pos:
                    components.append(mesh_pos)
                components.append(world_pos[t])
                components.append(node_type_oh)
                components.append(vel_normal[t])
                components.append(stress[t])
                feats_t = np.concatenate(components, axis=-1)
                feats_list.append(feats_t)

            # [FIX 1] Stack to (T, N, F) then permute to (N, T, F)
            X_seq = torch.tensor(np.stack(feats_list, axis=0), dtype=torch.float32)
            X_seq = X_seq.permute(1, 0, 2)  # Shape becomes (N, T, F)

            # Edges
            edge_index_mesh = build_mesh_edge_index(mesh_cells)
            pos_t0 = torch.tensor(world_pos[0], dtype=torch.float32)
            edge_index_world = torch.empty((2, 0), dtype=torch.long)

            if add_world_edges == 'radius':
                edge_index_world = radius_graph(pos_t0, r=radius, loop=False)
            elif add_world_edges == 'k_neighb':
                edge_index_world = knn_graph(pos_t0, k=k_neighb, loop=False)

            full_edge_index = torch.cat([edge_index_mesh, edge_index_world], dim=1)
            full_edge_index = coalesce(full_edge_index)

            # [FIX 2] Permute pos_seq as well
            pos_seq_tensor = torch.tensor(world_pos, dtype=torch.float32)
            pos_seq_tensor = pos_seq_tensor.permute(1, 0, 2)  # Shape becomes (N, T, D)

            data = Data(x=X_seq,
                        pos_seq=pos_seq_tensor,
                        edge_index=full_edge_index,
                        node_type=torch.tensor(node_type_idx, dtype=torch.long),
                        cells=torch.tensor(mesh_cells, dtype=torch.long))
            data_list.append(data)

        if not data_list:
            print("No data loaded. Check paths.")
            return

        # Compute Statistics
        print("Computing statistics for normalization...")
        sum_x = 0
        count = 0
        for data in data_list:
            # Sum over N (dim 0) and T (dim 1)
            sum_x += data.x.sum(dim=(0, 1))
            count += data.x.shape[0] * data.x.shape[1]
        mean = sum_x / count

        sum_diff_sq = 0
        for data in data_list:
            sum_diff_sq += ((data.x - mean) ** 2).sum(dim=(0, 1))
        std = torch.sqrt(sum_diff_sq / (count - 1))
        std[std < 1e-8] = 1.0

        # Custom Standard Normalization
        feat_idx = get_feature_indices(include_mesh_pos)
        if norm_method == "standard":
            max_std_pos = std[feat_idx.world_pos].max()
            std[feat_idx.world_pos] = max_std_pos
            max_std_vel = std[feat_idx.velocity].max()
            std[feat_idx.velocity] = max_std_vel
            mean[feat_idx.world_pos] = mean[feat_idx.world_pos].mean()
            mean[feat_idx.velocity] = mean[feat_idx.velocity].mean()

        print("Applying normalization...")
        for data in data_list:
            data.x = (data.x - mean) / std
            # Stats shape (1, 1, F) broadcasts fine to (N, T, F)
            data.stats_mean = mean.view(1, 1, -1)
            data.stats_std = std.view(1, 1, -1)

        # Save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Successfully saved dataset to {self.processed_paths[0]}")


# ==========================================
#  MAIN EXECUTION BLOCK (Run this to build)
# ==========================================
if __name__ == "__main__":
    # Load Config
    config_path = "pyg_dataconfig.yaml"  # Ensure this path is correct relative to execution
    if not os.path.exists(config_path):
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    dataset_root = config.get('output_dir')
    processed_dir = os.path.join(dataset_root)

    print("=================================================")
    print(f" DATA BUILDER: {dataset_root}")
    print("=================================================")

    # CLEANUP: Delete old processed file to force a rebuild
    # This ensures that running this script actually re-processes the data
    if os.path.exists(processed_dir):
        print(f"Cleaning old data in {processed_dir}...")
        shutil.rmtree(processed_dir)

    # Instantiate Dataset
    # This triggers init -> check files -> process() because files were deleted
    dataset = TrajectoryDataset(root=dataset_root, dataconfig=config, pre_transform=None, transform=None)
    path_print = os.path.join(processed_dir, 'pyg_processed_data.pt')
    print(f"Done! Data available at: {path_print}")