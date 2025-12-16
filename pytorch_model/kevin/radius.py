"""
Converted from `radius.ipynb`.
Demonstrates how to:
1) Load a deforming-plate trajectory
2) Set actuator (sphere) nodes to their t+1 positions
3) Build actuator->mesh world edges with a radius search
4) Optionally build all world edges with a radius graph (excluding actuator-actuator)
5) Visualize in 3D
"""

import torch
from pathlib import Path
import matplotlib.pyplot as plt
from torch_cluster import radius_graph, radius
from dataclasses import dataclass
from enum import IntEnum

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
# CHANGE THIS to point to your saved trajectories
DATASET_PATH = Path("datasets/deforming_plate/train.pth")
T_STEP = 100
NEAREST_NEIGHBOR_RADIUS = 0.03


# ----------------------------------------------------------------------
# Data structures
# ----------------------------------------------------------------------
@dataclass
class DeformingPlateSample:
    """One trajectory sample."""

    sample_idx: int
    mesh_pos: torch.Tensor
    world_pos: torch.Tensor
    cells: torch.Tensor
    node_type: torch.Tensor
    stress: torch.Tensor

    def get_timestep(self, time_idx: int) -> "DeformingPlateSample":
        return self.__class__(
            sample_idx=self.sample_idx,
            mesh_pos=self.mesh_pos,
            world_pos=self.world_pos[time_idx],
            cells=self.cells,
            node_type=self.node_type,
            stress=self.stress[time_idx],
        )


class MeshGraphNetNodeType(IntEnum):
    NORMAL_MESH_NODE = 0
    EXTERNAL_FORCE_MESH_NODE = 1  # actuator
    FIXED_MESH_NODE = 3


# ----------------------------------------------------------------------
# World-edge helpers
# ----------------------------------------------------------------------
def get_world_edges(sample: DeformingPlateSample, r: float = 0.03) -> torch.Tensor:
    """
    Build edges between actuator nodes (at t+1 position) and mesh nodes (at t)
    using a radius search.
    Returns edge_index [2, E] with bidirectional edges.
    """
    mesh_pos = sample.world_pos[sample.node_type != MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE]
    actuator_pos = sample.world_pos[sample.node_type == MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE]

    edge_radius_index = radius(mesh_pos, actuator_pos, r=r, max_num_neighbors=32)

    # Map back to original indices
    mesh_indices = torch.nonzero(sample.node_type != MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE, as_tuple=False).squeeze()
    actuator_indices = torch.nonzero(sample.node_type == MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE, as_tuple=False).squeeze()

    # radius returns edges y->x (actuator -> mesh)
    connected_actuator_indices = actuator_indices[edge_radius_index[0]]
    connected_mesh_indices = mesh_indices[edge_radius_index[1]]

    # Make edges undirected
    edge_index = torch.stack(
        [
            torch.cat([connected_mesh_indices, connected_actuator_indices]),
            torch.cat([connected_actuator_indices, connected_mesh_indices]),
        ],
        dim=0,
    )
    return edge_index


def get_gns_edges(node_types: torch.Tensor, world_pos: torch.Tensor, r: float = NEAREST_NEIGHBOR_RADIUS) -> torch.Tensor:
    """
    Build world edges among all nodes via radius graph, excluding actuator-actuator edges.
    Returns bidirectional edge_index [2, E].
    """
    world_edge_indices = radius_graph(world_pos, r=r, max_num_neighbors=32)
    actuator_actuator_mask = torch.logical_and(
        node_types[world_edge_indices[0]] == MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE,
        node_types[world_edge_indices[1]] == MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE,
    )
    edge_index = world_edge_indices[:, ~actuator_actuator_mask]
    full_edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    return full_edge_index


# ----------------------------------------------------------------------
# Visualization helpers
# ----------------------------------------------------------------------
def plot_edges(sample: DeformingPlateSample, edge_index: torch.Tensor, title: str):
    mesh = sample.world_pos[sample.node_type != MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE].numpy()
    actuator = sample.world_pos[sample.node_type == MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE].numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)

    ax.scatter(mesh[:, 0], mesh[:, 1], mesh[:, 2], label="mesh")
    ax.scatter(actuator[:, 0], actuator[:, 1], actuator[:, 2], label="actuator")

    for i in range(edge_index.shape[1]):
        p1 = sample.world_pos[edge_index[0, i]]
        p2 = sample.world_pos[edge_index[1, i]]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], color="gray", alpha=0.5)

    ax.legend()
    plt.show()


# ----------------------------------------------------------------------
# Main demo
# ----------------------------------------------------------------------
def main():
    # Load dataset and pick first trajectory
    raw_dataset = torch.load(DATASET_PATH)
    trajectory = DeformingPlateSample(**raw_dataset[0])

    # Get t and t+1 steps
    time_step = trajectory.get_timestep(T_STEP)
    time_step_plus_one = trajectory.get_timestep(T_STEP + 1)

    node_type = time_step.node_type
    # Set actuator nodes to their t+1 positions
    world_pos = time_step.world_pos.clone()
    world_pos[node_type == MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE] = (
        time_step_plus_one.world_pos[node_type == MeshGraphNetNodeType.EXTERNAL_FORCE_MESH_NODE]
    )
    time_step.world_pos = world_pos

    # Plot actuator->mesh world edges (radius on actuator->mesh only)
    edge_index = get_world_edges(time_step, r=NEAREST_NEIGHBOR_RADIUS)
    plot_edges(time_step, edge_index, title="Actuator->Mesh world edges (radius)")

    # Plot all world edges (radius graph, excluding actuator-actuator)
    full_edge_index = get_gns_edges(time_step.node_type, time_step.world_pos, r=NEAREST_NEIGHBOR_RADIUS)
    plot_edges(time_step, full_edge_index, title="All world edges (radius graph, no actuator-actuator)")


if __name__ == "__main__":
    main()
