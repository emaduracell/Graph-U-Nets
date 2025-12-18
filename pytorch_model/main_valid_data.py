import os
import torch
import yaml

from data_builder import load_all_trajectories_with_precomputed_stats
from helpers.helpers import load_config


def _preprocess_valid_and_save(dataconfig):
    """
    Loads validation data from TFRecord, recomputes world edges (as in train),
    and normalizes using train-fitted mean/std (loaded from preprocessed train metadata).
    """
    output_dir = dataconfig['output_dir']
    meta_path = dataconfig['meta_path']

    # Train stats path is derived from the *same* output_dir used for train preprocessing.
    train_metadata_path = os.path.join(output_dir, "preprocessed_metadata.pt")
    if not os.path.exists(train_metadata_path):
        raise FileNotFoundError(
            f"Train metadata not found at: {train_metadata_path}\n"
            f"Run `python main_data.py` first (or otherwise generate preprocessed train data+metadata)."
        )

    train_metadata = torch.load(train_metadata_path, map_location="cpu")
    if "mean" not in train_metadata or "std" not in train_metadata:
        raise KeyError(f"Train metadata at {train_metadata_path} must contain keys 'mean' and 'std'.")
    mean = train_metadata["mean"]
    std = train_metadata["std"]

    # Clone config and override TFRecord to validation split
    dataconfig_valid = dict(dataconfig)
    dataconfig_valid["tfrecord_path"] = "raw_data/valid.tfrecord"
    valid_max_trajs = dataconfig.get("valid_max_trajs")
    if valid_max_trajs is not None:
        dataconfig_valid["max_trajs"] = valid_max_trajs

    print("\n" + "=" * 60)
    print(" PREPROCESSING VALIDATION DATA")
    print("=" * 60 + "\n")
    print(f"  TFRecord (valid): {dataconfig_valid['tfrecord_path']}")
    print(f"  Meta: {meta_path}")
    print(f"  Max trajectories: {dataconfig_valid.get('max_trajs') if dataconfig_valid.get('max_trajs') else 'All'}")
    print(f"  Output directory: {output_dir}")
    print(f"  Train stats: {train_metadata_path}\n")

    list_of_trajs = load_all_trajectories_with_precomputed_stats(dataconfig_valid, mean, std)

    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, "preprocessed_valid.pt")
    torch.save(list_of_trajs, output_path)
    print(f"\n Saved {len(list_of_trajs)} preprocessed validation trajectories to: {output_path}")

    # Save validation metadata (includes the same train mean/std for convenience)
    if len(list_of_trajs) > 0:
        sample_traj = list_of_trajs[0]
        metadata = {
            "num_trajectories": len(list_of_trajs),
            "feature_dim": sample_traj["X_seq_norm"].shape[2],
            "time_steps": sample_traj["X_seq_norm"].shape[0],
            "num_nodes": sample_traj["X_seq_norm"].shape[1],
            "mean": mean,
            "std": std,
            "tfrecord_path": dataconfig_valid["tfrecord_path"],
            "meta_path": meta_path,
            "train_metadata_path": train_metadata_path,
        }
        metadata_path = os.path.join(output_dir, "preprocessed_valid_metadata.pt")
        torch.save(metadata, metadata_path)
        print(f" Saved validation metadata to: {metadata_path}")

    # Save used dataconfig for traceability
    used_cfg_path = os.path.join(output_dir, "used_dataconfig_valid.yaml")
    with open(used_cfg_path, "w") as f:
        yaml.safe_dump(dataconfig_valid, f, sort_keys=False)
    print(f" Saved used validation dataconfig to: {used_cfg_path}")

    return output_path


def main(dataconfig):
    _preprocess_valid_and_save(dataconfig)


if __name__ == "__main__":
    dataconfig_path = os.path.join(os.path.dirname(__file__), "dataconfig.yaml")
    dataconfig = load_config(dataconfig_path)['data']
    main(dataconfig)


