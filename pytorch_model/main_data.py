import torch
import os
from data_loader import load_all_trajectories, load_config

def preprocess_and_save(tfrecord_path, meta_path, output_dir, max_trajs, mesh_pos_idxs, world_pos_idxs,
                        node_type_idxs, velocity_idxs, stress_idxs, include_mesh_pos, norm_method):
    """
    Loads data using data_loader.py, then saves torch files for data and metadata in proper directory

    :param tfrecord_path:
    :param meta_path:
    :param output_dir:
    :param max_trajs:
    :param mesh_pos_idxs:
    :param world_pos_idxs:
    :param node_type_idxs:
    :param velocity_idxs:
    :param stress_idxs:
    :param include_mesh_pos:
    :param norm_method:

    :return:
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
    list_of_trajs = load_all_trajectories(tfrecord_path, meta_path, max_trajs, mesh_pos_idxs, world_pos_idxs,
                                          node_type_idxs, velocity_idxs, stress_idxs, include_mesh_pos,
                                          norm_method)

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

    return output_path


def main(tfrecord_path, meta_path, max_trajs, output_dir, mesh_pos_idxs, world_pos_idxs, node_type_idxs, velocity_idxs,
         stress_idxs, include_mesh_pos, norm_method):
    """
    access point function to generate data

    :param tfrecord_path:
    :param meta_path:
    :param max_trajs:
    :param output_dir:
    :param mesh_pos_idxs:
    :param world_pos_idxs:
    :param node_type_idxs:
    :param velocity_idxs:
    :param stress_idxs:
    :param include_mesh_pos:
    :param norm_method:

    :return: nothing
    """
    preprocess_and_save(tfrecord_path, meta_path, output_dir, max_trajs, mesh_pos_idxs, world_pos_idxs,
                        node_type_idxs, velocity_idxs, stress_idxs, include_mesh_pos, norm_method)


if __name__ == "__main__":
    dataconfig_path = os.path.join(os.path.dirname(__file__), "dataconfig.yaml")
    dataconfig = load_config(dataconfig_path)
    include_mesh_pos = dataconfig['data']['include_mesh_pos']
    norm_method = dataconfig['data']['normalization_method']
    max_trajs = dataconfig['data']['max_trajs']
    tfrecord_path = dataconfig['data']['tfrecord_path']
    meta_path = dataconfig['data']['meta_path']
    output_dir = dataconfig['data']['output_dir']
    output_dir = output_dir + f"_{norm_method}_{include_mesh_pos}"

    if norm_method not in ['centroid', 'standard']:
        raise ValueError(f"norm_method == {norm_method} not supported")

    if include_mesh_pos:
        mesh_pos_idxs = slice(0, 3)
        world_pos_idxs = slice(3, 6)
        node_type_idxs = slice(6, 8)
        velocity_idxs = slice(8, 11)
        stress_idxs = slice(11, 12)
        dim_in = 12  # mesh_pos (3) + world_pos (3) + node_type (2) + vel (3) + stress (1)
    else:
        world_pos_idxs = slice(0, 3)
        node_type_idxs = slice(3, 5)
        velocity_idxs = slice(5, 8)
        stress_idxs = slice(8, 9)
        mesh_pos_idxs = None
        dim_in = 9 # world_pos (3) + node_type (2) + vel (3) + stress (1)

    main(tfrecord_path, meta_path, max_trajs, output_dir, mesh_pos_idxs, world_pos_idxs, node_type_idxs, velocity_idxs,
         stress_idxs, include_mesh_pos, norm_method)
