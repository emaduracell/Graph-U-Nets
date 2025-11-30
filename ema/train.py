import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import os

from data_loader import load_all_trajectories
from datasetclass import EmaUnetDataset, collate_ema_unet
from graph_unet_defplate import GNet_EMA

# HARDCODED DATASE  T AND OUTPUT PATHS
TFRECORD_PATH = "data/train.tfrecord"
META_PATH = "data/meta.json"
NUM_TRAIN_TRAJS = 1  # Load only the first K trajectories
CHECKPOINT_PATH = "old_models/gnet_ema_multi_prev.pt"


def load_config(config_path):
    """Load model and training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_gnet_ema(device):
    """Training loop"""
    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    # Extract model and training parameters
    model_cfg = config['model']
    train_cfg = config['training']
    # load model configs from yaml
    model_hyperparams = lambda: None
    model_hyperparams.activation_gnn = model_cfg['activation_gnn']
    model_hyperparams.activation_mlps_final = model_cfg['activation_mlps_final']
    model_hyperparams.hid_gnn_layer_dim = model_cfg['hid_gnn_layer_dim']
    model_hyperparams.hid_mlp_dim = model_cfg['hid_mlp_dim']
    model_hyperparams.k_pool_ratios = model_cfg['k_pool_ratios']
    model_hyperparams.dropout_gnn = model_cfg['dropout_gnn']
    model_hyperparams.dropout_mlps_final = model_cfg['dropout_mlps_final']

    print("\n=================================================")
    print(" LOADING TRAJECTORIES")
    print("=================================================\n")
    print(f"TFRecord: {TFRECORD_PATH}")
    print(f"Meta: {META_PATH}")
    print(f"Max trajectories: {NUM_TRAIN_TRAJS if NUM_TRAIN_TRAJS else 'All'}")
    # load trajectories (only first K if specified)
    list_of_trajs = load_all_trajectories(TFRECORD_PATH, META_PATH, max_trajs=NUM_TRAIN_TRAJS)

    # Build dataset from these trajectories
    dataset = EmaUnetDataset(list_of_trajs)
    print(f"Total training pairs X_t --> X_t+1: {len(dataset)}")
    # Random 80/20 split and then load data
    total = len(dataset)
    perm = torch.randperm(total)
    split = int(0.8 * total)
    train_idx = perm[:split]
    test_idx = perm[split:]
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)
    train_loader = DataLoader(train_set, batch_size=train_cfg['batch_size'], shuffle=train_cfg['shuffle'],
                              collate_fn=collate_ema_unet)
    test_loader = DataLoader(test_set, batch_size=train_cfg['batch_size'], shuffle=False, collate_fn=collate_ema_unet)

    # Build model
    dim_in = list_of_trajs[0]["X_seq_norm"].shape[2]
    dim_out_vel = 3
    dim_out_stress = 1
    model = GNet_EMA(dim_in, dim_out_vel, dim_out_stress, model_hyperparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    loss_fn = torch.nn.MSELoss()

    # Training loop
    print("\n=================================================")
    print(" TRAINING")
    print("=================================================\n")
    print(f"Epochs: {train_cfg['epochs']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Learning rate: {train_cfg['lr']}\n")

    for epoch in range(train_cfg['epochs']):

        # Train phase
        model.train()
        total_train_loss = 0.0
        for adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types, traj_ids in train_loader:
            adj_mat_list = [A.to(device) for A in adj_mat_list]
            feat_t_mat_list = [X_t.to(device) for X_t in feat_t_mat_list]
            feat_tp1_mat_list = [X_tp1.to(device) for X_tp1 in feat_tp1_mat_list]
            node_types = [nt.to(device) for nt in node_types]
            # One epoch
            batch_loss, preds_list = model(adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, node_types)
            # backprop
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # test loss
            total_train_loss += batch_loss.item()

        # Evaluation
        model.eval()
        total_test_loss = 0.0
        with torch.no_grad():
            for adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types, traj_ids in test_loader:
                gs = [A.to(device) for A in adj_mat_list]
                hs = [X_t.to(device) for X_t in feat_t_mat_list]
                targets = [X_tp1.to(device) for X_tp1 in feat_tp1_mat_list]
                node_types = [nt.to(device) for nt in node_types]

                batch_loss, preds_list = model(gs, hs, targets, node_types)
                total_test_loss += batch_loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_test = total_test_loss / len(test_loader)

        print(f"[Epoch {epoch:03d}]  Train Loss: {avg_train:.6f}  |  Test Loss: {avg_test:.6f}")

    print(f"\nSaving model to {CHECKPOINT_PATH}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    # TODO ADD PLOTs of


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_gnet_ema(device)
