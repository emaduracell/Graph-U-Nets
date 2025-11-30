import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import yaml
import os

from ema_data_loader_multi import load_all_trajectories
from ema_datasetclass import EmaUnetDataset, collate_ema_unet
from ema_unet import GNet_EMA

# HARDCODED DATASET AND OUTPUT PATHS
TFRECORD_PATH = "data/train.tfrecord"
META_PATH = "data/meta.json"
NUM_TRAIN_TRAJS = 3  # Load only the first K trajectories
CHECKPOINT_PATH = "gnet_ema_multi.pt"

# CONFIGURATION LOADING

def load_config(config_path="config.yaml"):
    """Load model and training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# ------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------

def train_gnet_ema():

    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    
    # Extract model and training parameters
    model_cfg = config['model']
    train_cfg = config['training']
    
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("\n=================================================")
    print(" LOADING TRAJECTORIES")
    print("=================================================\n")
    print(f"TFRecord: {TFRECORD_PATH}")
    print(f"Meta: {META_PATH}")
    print(f"Max trajectories: {NUM_TRAIN_TRAJS if NUM_TRAIN_TRAJS else 'All'}")

    # load trajectories (only first K if specified)
    list_of_trajs = load_all_trajectories(
        TFRECORD_PATH,
        META_PATH,
        max_trajs=NUM_TRAIN_TRAJS
    )

    # build dataset from these trajectories
    dataset = EmaUnetDataset(list_of_trajs)

    print(f"Total training pairs X_t→X_t+1: {len(dataset)}")
    # RANDOM 80/20 SAMPLE SPLIT
    total = len(dataset)
    perm = torch.randperm(total)
    split = int(0.8 * total)
    train_idx = perm[:split]
    test_idx  = perm[split:]

    train_set = Subset(dataset, train_idx)
    test_set  = Subset(dataset, test_idx)

    train_loader = DataLoader(
        train_set,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        collate_fn=collate_ema_unet
    )

    test_loader = DataLoader(
        test_set,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        collate_fn=collate_ema_unet
    )
    # BUILD MODEL
    F_in = list_of_trajs[0]["X_seq_norm"].shape[2]  # feature dimension (8: pos + node_type + vel + stress)
    F_out = 4  # output dimension (3 velocity + 1 stress)

    myargs = lambda: None
    myargs.act_n = model_cfg['act_n']
    myargs.act_c = model_cfg['act_c']
    myargs.l_dim = model_cfg['l_dim']
    myargs.h_dim = model_cfg['h_dim']
    myargs.ks = model_cfg['ks']
    myargs.drop_n = model_cfg['drop_n']
    myargs.drop_c = model_cfg['drop_c']

    model = GNet_EMA(F_in, F_out, myargs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'])
    loss_fn = torch.nn.MSELoss()

    # TRAINING LOOP
    print("\n=================================================")
    print(" TRAINING")
    print("=================================================\n")
    print(f"Epochs: {train_cfg['epochs']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Learning rate: {train_cfg['lr']}\n")

    for epoch in range(train_cfg['epochs']):

        # -------- TRAIN --------
        model.train()
        total_train_loss = 0.0

        for As, X_ts, X_tp1s, means, stds, cells, node_types, traj_ids in train_loader:

            batch_loss = 0.0

            # compute gradients graph by graph
            for A, X_t, X_tp1, node_type in zip(As, X_ts, X_tp1s, node_types):

                A = A.to(device)
                X_t = X_t.to(device)
                X_tp1 = X_tp1.to(device)
                node_type = node_type.to(device)  # [N], integer node types

                pred = model.rollout_step(A, X_t)   # [N, 4] (velocity + stress)
                
                # Create masks for filtering
                # Velocity: only node_type == 0
                vel_mask = (node_type == 0)  # [N]
                # Stress: node_type == 0 or node_type == 6
                stress_mask = (node_type == 0) | (node_type == 6)  # [N]

                # Extract targets
                target_vel = X_tp1[:, 4:7]      # [N, 3]
                target_stress = X_tp1[:, 7:8]   # [N, 1]
                
                # Extract predictions
                pred_vel = pred[:, :3]          # [N, 3]
                pred_stress = pred[:, 3:4]      # [N, 1]
                
                # Compute separate losses with masks
                loss = 0.0
                if vel_mask.sum() > 0:
                    loss_vel = loss_fn(pred_vel[vel_mask], target_vel[vel_mask])
                    loss = loss + loss_vel
                
                if stress_mask.sum() > 0:
                    loss_stress = loss_fn(pred_stress[stress_mask], target_stress[stress_mask])
                    loss = loss + loss_stress

                batch_loss += loss

            # average loss over graphs in batch
            batch_loss = batch_loss / len(As)

            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

            total_train_loss += batch_loss.item()

        # -------- TEST --------
        model.eval()
        total_test_loss = 0.0

        with torch.no_grad():
            for As, X_ts, X_tp1s, means, stds, cells, node_types, traj_ids in test_loader:

                batch_loss = 0.0

                for A, X_t, X_tp1, node_type in zip(As, X_ts, X_tp1s, node_types):
                    A = A.to(device)
                    X_t = X_t.to(device)
                    X_tp1 = X_tp1.to(device)
                    node_type = node_type.to(device)  # [N]

                    pred = model.rollout_step(A, X_t)  # [N, 4] (velocity + stress)
                    
                    # Create masks for filtering
                    # Velocity: only node_type == 0
                    vel_mask = (node_type == 0)  # [N]
                    # Stress: node_type == 0 or node_type == 6
                    stress_mask = (node_type == 0) | (node_type == 6)  # [N]
                    
                    # Extract targets
                    target_vel = X_tp1[:, 4:7]      # [N, 3]
                    target_stress = X_tp1[:, 7:8]   # [N, 1]
                    
                    # Extract predictions
                    pred_vel = pred[:, :3]          # [N, 3]
                    pred_stress = pred[:, 3:4]      # [N, 1]
                    
                    # Compute separate losses with masks
                    loss = 0.0
                    if vel_mask.sum() > 0:
                        loss_vel = loss_fn(pred_vel[vel_mask], target_vel[vel_mask])
                        loss = loss + loss_vel
                    
                    if stress_mask.sum() > 0:
                        loss_stress = loss_fn(pred_stress[stress_mask], target_stress[stress_mask])
                        loss = loss + loss_stress

                    batch_loss += loss

                batch_loss = batch_loss / len(As)
                total_test_loss += batch_loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_test  = total_test_loss / len(test_loader)

        print(f"[Epoch {epoch:03d}]  Train Loss: {avg_train:.6f}  |  Test Loss: {avg_test:.6f}")

    # SAVE CHECKPOINT
    
    print(f"\nSaving model to {CHECKPOINT_PATH}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)


if __name__ == "__main__":
    train_gnet_ema()
