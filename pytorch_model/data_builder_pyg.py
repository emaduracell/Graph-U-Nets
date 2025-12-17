import os
import json
import torch
import numpy as np
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
    """
    Vectorized conversion of mesh cells (quads) to COO edge_index.
    """
    cells = torch.tensor(mesh_cells, dtype=torch.long)

    # Define the 6 edges for a fully connected quad
    # Source nodes
    s = torch.cat([
        cells[:, 0], cells[:, 0], cells[:, 0],
        cells[:, 1], cells[:, 1],
        cells[:, 2]
    ])
    # Target nodes
    t = torch.cat([
        cells[:, 1], cells[:, 2], cells[:, 3],
        cells[:, 2], cells[:, 3],
        cells[:, 3]
    ])

    edge_index = torch.stack([s, t], dim=0)

    # Make undirected and remove duplicates
    edge_index = to_undirected(edge_index)

    return edge_index


def build_velocity(world_pos, mode):
    if mode not in ["normal", "actuator"]:
        raise ValueError(f"Unknown mode = {mode}")

    # Input world_pos is (T, N, 3)
    vel = np.zeros_like(world_pos)

    if mode == "normal":
        vel[1:] = world_pos[1:] - world_pos[:-1]
    elif mode == "actuator":
        vel[1:-1] = world_pos[2:] - world_pos[1:-1]

    return vel


class TrajectoryDataset(InMemoryDataset):

    def __init__(self, root, dataconfig, transform=None, pre_transform=None):
        self.dataconfig = dataconfig
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return []  # We load from TFRecord path in config, not raw_dir

    @property
    def processed_file_names(self):
        return ['pyg_processed_data.pt']

    def process(self):
        print("Processing data from TFRecords to PyG Data objects...")

        # Config loading
        tfrecord_path = self.dataconfig['tfrecord_path']
        meta_path = self.dataconfig['meta_path']
        max_trajs = self.dataconfig['max_trajs']
        include_mesh_pos = self.dataconfig['include_mesh_pos']
        norm_method = self.dataconfig['normalization_method']

        # World Edge Config
        add_world_edges = self.dataconfig['add_world_edges']
        radius = self.dataconfig['radius_world_edge']
        k_neighb = self.dataconfig['k_neighb']
        a_time_var = self.dataconfig.get('a_time_var', False)

        with open(meta_path, "r") as f:
            meta = json.load(f)

        loader = tfrecord_loader(tfrecord_path, index_path=None)

        data_list = []

        # Create data objects
        for i, record in enumerate(loader):
            if max_trajs is not None and i >= max_trajs:
                break

            traj = cast_trajectory_from_record(record, meta)

            # 1. Extract Raw Data
            world_pos = traj["world_pos"]  # (T, N, 3)
            node_type = traj["node_type"]  # (N, 1)
            stress = traj["stress"]  # (T, N, 1)
            mesh_cells = traj["cells"]  # (C, 4)
            mesh_pos = traj["mesh_pos"] if include_mesh_pos else None

            T, N, _ = world_pos.shape

            # 2. Compute Velocities
            vel_normal = build_velocity(world_pos, "normal")
            vel_actuator = build_velocity(world_pos, "actuator")

            # Fix actuator velocity logic
            actuator_mask = (node_type == SPHERE_NODE).reshape(-1)
            vel_normal[:, actuator_mask, :] = vel_actuator[:, actuator_mask, :]

            # 3. Node Types One-Hot
            lookup = np.array([NORMAL_NODE_OH, SPHERE_NODE_OH, [0, 0], BOUNDARY_NODE_OH])
            node_type_idx = node_type.squeeze(-1)
            node_type_oh = lookup[node_type_idx].astype(np.float32)

            # 4. Build Feature Sequence X (T, N, F) [mesh_pos?, pos, node_type, vel, stress]
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

            X_seq = torch.tensor(np.stack(feats_list, axis=0), dtype=torch.float32)

            # 5. Build Sparse Edge Index (Static Mesh)
            edge_index_mesh = build_mesh_edge_index(mesh_cells)

            # 6. Add Initial/Static World Edges
            pos_t0 = torch.tensor(world_pos[0], dtype=torch.float32)

            edge_index_world = torch.empty((2, 0), dtype=torch.long)

            if add_world_edges == 'radius':
                edge_index_world = radius_graph(pos_t0, r=radius, loop=False)

            elif add_world_edges == 'k_neighb':
                edge_index_world = knn_graph(pos_t0, k=k_neighb, loop=False)

            # Combine Mesh and World edges
            full_edge_index = torch.cat([edge_index_mesh, edge_index_world], dim=1)
            full_edge_index = coalesce(full_edge_index)  # Remove duplicates and sort

            # 7. Create Data Object
            data = Data(x=X_seq,
                pos_seq=torch.tensor(world_pos, dtype=torch.float32),  # (T, N, 3)
                edge_index=full_edge_index,  # (2, E_static)
                node_type=torch.tensor(node_type_idx, dtype=torch.long),
                cells=torch.tensor(mesh_cells, dtype=torch.long)
            )
            data_list.append(data)

        if not data_list:
            print("No data loaded.")
            return

        # Pass 2
        # Compute global mean and std
        print("Computing statistics...")

        # Accumulate sum
        sum_x = 0
        count = 0
        for data in data_list:
            # collapse T and N
            sum_x += data.x.sum(dim=(0, 1))
            count += data.x.shape[0] * data.x.shape[1]

        mean = sum_x / count

        # Accumulate std
        sum_diff_sq = 0
        for data in data_list:
            sum_diff_sq += ((data.x - mean) ** 2).sum(dim=(0, 1))

        std = torch.sqrt(sum_diff_sq / (count - 1))

        # Handle constants (avoid div by zero)
        std[std < 1e-8] = 1.0

        # Standard normalization logic from original code (override pos/vel stats)
        feat_idx = get_feature_indices(include_mesh_pos)

        if norm_method == "standard":
            # Average std across spatial components for Pos
            max_std_pos = std[feat_idx.world_pos].max()
            std[feat_idx.world_pos] = max_std_pos

            max_std_vel = std[feat_idx.velocity].max()
            std[feat_idx.velocity] = max_std_vel

            # Center Pos and Vel means? (Usually we want 0 mean for vel, center for pos)
            # Keeping original logic roughly:
            mean[feat_idx.world_pos] = mean[feat_idx.world_pos].mean()
            mean[feat_idx.velocity] = mean[feat_idx.velocity].mean()

        # Apply Normalization
        print("Applying normalization...")
        for data in data_list:
            data.x = (data.x - mean) / std
            # Save stats for un-normalizing later
            data.stats_mean = mean.view(1, 1, -1)
            data.stats_std = std.view(1, 1, -1)

        # Save
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Saved dataset to {self.processed_paths[0]}")