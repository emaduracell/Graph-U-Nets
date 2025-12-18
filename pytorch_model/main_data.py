import torch
import os
from data_builder import load_all_trajectories
from helpers.helpers import load_config
import yaml

def _preprocess_and_save(dataconfig):
    """
    Loads data using data_loader.py, then saves torch files for data and metadata in proper directory

    Args:
        dataconfig
    """
    tfrecord_path = dataconfig['tfrecord_path']
    max_trajs = dataconfig['max_trajs']
    output_dir = dataconfig['output_dir']
    meta_path = dataconfig['meta_path']

    print("\n" + "=" * 60)
    print(" PREPROCESSING DATA")
    print("=" * 60 + "\n")
    print(f"  TFRecord: {tfrecord_path}")
    print(f"  Meta: {meta_path}")
    print(f"  Max trajectories: {max_trajs if max_trajs else 'All'}")
    print(f"  Output directory: {output_dir}\n")

    # Load and preprocess all trajectories. Trajectories are loaded in sequential order.
    list_of_trajs = load_all_trajectories(dataconfig)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    # Save preprocessed trajectories
    output_path = os.path.join(output_dir, "preprocessed_train.pt")
    torch.save(list_of_trajs, output_path)
    print(f"\n Saved {len(list_of_trajs)} preprocessed trajectories to: {output_path}")

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
        print(f" Saved metadata to: {metadata_path}")

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

    # Save used dataconfig (identical to the one passed in)
    used_cfg_path = os.path.join(output_dir, "used_dataconfig.yaml")
    with open(used_cfg_path, "w") as f:
        yaml.safe_dump(dataconfig, f, sort_keys=False)
    print(f" Saved used dataconfig to: {used_cfg_path}")

    return output_path


def main(dataconfig):
    """
    Access point function to generate data

    Args:
        dataconfig: config.yaml
    """
    _preprocess_and_save(dataconfig)


if __name__ == "__main__":
    dataconfig_path = os.path.join(os.path.dirname(__file__), "dataconfig.yaml")
    dataconfig = load_config(dataconfig_path)['data']
    main(dataconfig)