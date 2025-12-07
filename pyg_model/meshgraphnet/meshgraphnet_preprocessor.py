import os
import torch
from tqdm import tqdm
import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from physicsnemo.distributed.manager import DistributedManager

from deforming_plate_dataset import DeformingPlateDataset
from helpers import add_world_edges


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig):
    """
    Creates a distributed preprocessor

    Split the samples among ranks (note ranks = process so this is only concerning distributed training).
    Uses DeformingPlateDataset to iterate over all samples and all time steps, calling dataset[idx] to get the
    per-time-step graphs and adding world edges for each of them.
    - Because: the mesh connectivity (cells/edges) is static
    - node positions world_pos change in time --> thus Edge features (displacements, norms) are functions of
        these positions, so they are time-varying.

    For each sample:
    1. graph = daatset(idx) sarà node features + adjacency matrix + world position
    2. build mesh/world_EDGE_features
    3. create sample object with graph+mesh/world_EDGE_feat
    """

    # Initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    # Set up output directory
    output_dir = to_absolute_path(cfg.preprocess_output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    dataset = DeformingPlateDataset(
        name="deforming_plate_train",
        data_dir=to_absolute_path(cfg.data_dir),
        split="train",
        num_samples=cfg.num_training_samples,
        num_steps=cfg.num_training_time_steps,
    )

    num_samples = cfg.num_training_samples
    num_steps = cfg.num_training_time_steps

    # Split the samples among ranks (note ranks = process so this is only concerning distributed training)
    # This is used only to calcolate start, end of tqdm
    per_rank = num_samples // dist.world_size
    start = dist.rank * per_rank
    end = (
        (dist.rank + 1) * per_rank if dist.rank != dist.world_size - 1 else num_samples
    )

    # For each sample:
    # 1. graph = dataset(idx) sarà node features + adjacency matrix + world position
    # 2. build mesh/world_EDGE_features
    # 3. create sample with graph+mesh/world_EDGE_feat
    for sample_idx in tqdm(range(start, end), desc=f"Rank {dist.rank} preprocessing"):
        sample_file = os.path.join(output_dir, f"sample_{sample_idx:05d}.pt")
        if os.path.exists(sample_file):
            continue  # Skip if already processed

        sample_data = []
        for t in range(num_steps - 1):
            idx = sample_idx * (num_steps - 1) + t
            graph = dataset[idx].to(dist.device)
            graph, mesh_edge_features, world_edge_features = add_world_edges(graph)
            sample_data.append(
                {
                    "graph": graph,
                    "mesh_edge_features": mesh_edge_features,
                    "world_edge_features": world_edge_features,
                }
            )
        torch.save(sample_data, sample_file)
    print(f"Rank {dist.rank} finished processing samples {start} to {end - 1}")


if __name__ == "__main__":
    main()
