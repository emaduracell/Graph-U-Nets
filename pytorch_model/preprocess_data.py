import torch
import os
from data_loader import load_all_trajectories  # TODO CHANGE

# Global configuration constants
TFRECORD_PATH = "data/train.tfrecord"
META_PATH = "data/meta.json"
OUTPUT_DIR = "data"
MAX_TRAJS = None  # Set to None to load all trajectories, or specify a number


def preprocess_and_save(tfrecord_path, meta_path, output_dir, max_trajs):
    """
    Load trajectories from TFRecord, preprocess them, and save to disk.
    
    :param tfrecord_path: Path to the TFRecord file
    :param meta_path: Path to meta.json
    :param output_dir: Directory to save preprocessed data
    :param max_trajs: Maximum number of trajectories to load (None for all)
    """
    print("\n" + "=" * 60)
    print(" PREPROCESSING DATA")
    print("=" * 60 + "\n")
    print(f"  TFRecord: {tfrecord_path}")
    print(f"  Meta: {meta_path}")
    print(f"  Max trajectories: {max_trajs if max_trajs else 'All'}")
    print(f"  Output directory: {output_dir}\n")

    # Load and preprocess all trajectories
    # Note: Trajectories are loaded in deterministic sequential order from TFRecord
    # and will be saved maintaining this order (traj_id 0, 1, 2, ...)
    list_of_trajs = load_all_trajectories(tfrecord_path, meta_path, max_trajs=max_trajs)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save preprocessed trajectories
    output_path = os.path.join(output_dir, "preprocessed_train.pt")
    torch.save(list_of_trajs, output_path)
    print(f"\n✓ Saved {len(list_of_trajs)} preprocessed trajectories to: {output_path}")

    # Save metadata for reference
    if len(list_of_trajs) > 0:
        sample_traj = list_of_trajs[0]
        metadata = {
            "num_trajectories": len(list_of_trajs),
            "feature_dim": sample_traj["X_seq_norm"].shape[2],
            "time_steps": sample_traj["X_seq_norm"].shape[0],
            "num_nodes": sample_traj["X_seq_norm"].shape[1],
            "mean": sample_traj["mean"],
            "std": sample_traj["std"],
            "tfrecord_path": tfrecord_path,
            "meta_path": meta_path,
        }
        metadata_path = os.path.join(output_dir, "preprocessed_metadata.pt")
        torch.save(metadata, metadata_path)
        print(f"✓ Saved metadata to: {metadata_path}")

        print("\n" + "=" * 60)
        print(" DATASET SUMMARY")
        print("=" * 60)
        print(f"  Total trajectories: {metadata['num_trajectories']}")
        print(f"  Time steps per trajectory: {metadata['time_steps']}")
        print(f"  Nodes per trajectory: {metadata['num_nodes']}")
        print(f"  Feature dimension: {metadata['feature_dim']}")
        print(f"  Total training pairs: {metadata['num_trajectories'] * (metadata['time_steps'] - 1)}")
        print(f"  Note: Trajectories are saved in sequential order (0, 1, 2, ...)")
        print("=" * 60 + "\n")

    return output_path


def main():
    """Run preprocessing with global configuration constants."""
    preprocess_and_save(
        tfrecord_path=TFRECORD_PATH,
        meta_path=META_PATH,
        output_dir=OUTPUT_DIR,
        max_trajs=MAX_TRAJS
    )


if __name__ == "__main__":
    main()
