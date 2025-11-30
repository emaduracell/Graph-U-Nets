import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import argparse

from ema_data_loader_multi import load_all_trajectories
from ema_datasetclass import EmaUnetDataset, collate_ema_unet
from ema_unet import GNet_EMA

# ARGUMENT PARSING

def get_args():
    parser = argparse.ArgumentParser()

    # dataset paths
    parser.add_argument("--tfrecord", type=str,
                        default="deforming_plate/train.tfrecord")
    parser.add_argument("--meta", type=str,
                        default="deforming_plate/meta.json")

    # choose how many trajectories to load
    parser.add_argument("--num_train_trajs", type=int, default=None,
                        help="Load only the first K trajectories from TFRecord.")

    # model hyperparams
    parser.add_argument("--act_n", type=str, default="ELU")
    parser.add_argument("--act_c", type=str, default="ELU")
    parser.add_argument("--l_dim", type=int, default=128)
    parser.add_argument("--h_dim", type=int, default=256)
    parser.add_argument("--ks", nargs="+", type=float, default=[0.9, 0.8, 0.7])
    parser.add_argument("--drop_n", type=float, default=0.1)
    parser.add_argument("--drop_c", type=float, default=0.1)

    # training
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=4)

    # output
    parser.add_argument("--checkpoint", type=str,
                        default="gnet_ema_multi.pt")

    return parser.parse_args()


# ------------------------------------------------------------
# TRAINING LOOP
# ------------------------------------------------------------

def train_gnet_ema():

    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("\n=================================================")
    print(" LOADING TRAJECTORIES")
    print("=================================================\n")

    # load trajectories (only first K if specified)
    list_of_trajs = load_all_trajectories(
        args.tfrecord,
        args.meta,
        max_trajs=args.num_train_trajs
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
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_ema_unet
    )

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_ema_unet
    )
    # BUILD MODEL
    F_in = list_of_trajs[0]["X_seq_norm"].shape[2]  # feature dimension (8: pos + node_type + vel + stress)
    F_out = 4  # output dimension (3 velocity + 1 stress)

    myargs = lambda: None
    myargs.act_n = args.act_n
    myargs.act_c = args.act_c
    myargs.l_dim = args.l_dim
    myargs.h_dim = args.h_dim
    myargs.ks = args.ks
    myargs.drop_n = args.drop_n
    myargs.drop_c = args.drop_c

    model = GNet_EMA(F_in, F_out, myargs).to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.MSELoss()

    # TRAINING LOOP
    print("\n=================================================")
    print(" TRAINING")
    print("=================================================\n")

    for epoch in range(args.epochs):

        # -------- TRAIN --------
        model.train()
        total_train_loss = 0.0

        for As, X_ts, X_tp1s, means, stds, cells, traj_ids in train_loader:

            batch_loss = 0.0

            # compute gradients graph by graph
            for A, X_t, X_tp1 in zip(As, X_ts, X_tp1s):

                A = A.to(device)
                X_t = X_t.to(device)
                X_tp1 = X_tp1.to(device)

                pred = model.rollout_step(A, X_t)   # [N, 4] (velocity + stress)
                
                # Extract node_type from input features (feature index 3)
                node_type = X_t[:, 3]  # [N]
                
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
            for As, X_ts, X_tp1s, means, stds, cells, traj_ids in test_loader:

                batch_loss = 0.0

                for A, X_t, X_tp1 in zip(As, X_ts, X_tp1s):
                    A = A.to(device)
                    X_t = X_t.to(device)
                    X_tp1 = X_tp1.to(device)

                    pred = model.rollout_step(A, X_t)  # [N, 4] (velocity + stress)
                    
                    # Extract node_type from input features (feature index 3)
                    node_type = X_t[:, 3]  # [N]
                    
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
    
    print(f"\nSaving model to {args.checkpoint}")
    torch.save(model.state_dict(), args.checkpoint)


if __name__ == "__main__":
    train_gnet_ema()
