import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml
import os
import numpy as np
import sys

# Ensure we can import from the current directory
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_all_trajectories
from datasetclass import EmaUnetDataset, collate_ema_unet
from graph_unet_defplate import GNet_EMA
from plots import make_final_plots

# HARDCODED DATASE  T AND OUTPUT PATHS
TFRECORD_PATH = "data/train.tfrecord"
META_PATH = "data/meta.json"
NUM_TRAIN_TRAJS = 6  # Load only the first K trajectories
CHECKPOINT_PATH = "gnet_ema_multi.pt"
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def load_config(config_path):
    """Load model and training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_grad_norm(model):
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_batch_metrics(preds_list, targets_list, node_types_list):
    """Compute MAE for the batch."""
    total_mae = 0.0
    count = 0
    
    for pred, target, nt in zip(preds_list, targets_list, node_types_list):
        vel_mask = (nt == 0)
        stress_mask = (nt == 0) | (nt == 6)
        
        target_vel = target[:, 4:7]
        target_stress = target[:, 7:8]
        pred_vel = pred[:, :3]
        pred_stress = pred[:, 3:4]
        
        mae_graph = 0.0
        if vel_mask.any():
            mae_graph += F.l1_loss(pred_vel[vel_mask], target_vel[vel_mask], reduction='sum').item()
            count += vel_mask.sum().item() * 3
        if stress_mask.any():
            mae_graph += F.l1_loss(pred_stress[stress_mask], target_stress[stress_mask], reduction='sum').item()
            count += stress_mask.sum().item() * 1 #
            
        total_mae += mae_graph

    return total_mae, count


def run_final_evaluation(model, test_loader, device, train_losses, val_losses, train_maes, val_maes, grad_norms):
    """
    Runs the final evaluation loop, collects data, and calls the plotting function.
    """
    print("Generating final evaluation plots...")
    
    # activations (using a hook on one batch, in particular im keeping only the last graph)
    activations = {}
    # Hook into the velocity_mlp input to get latent features
    handle = model.velocity_mlp.register_forward_hook(
        lambda m, i, o: activations.update({'latent_features': i[0].detach().cpu().numpy()})
    )
    
    all_vel_preds = []
    all_vel_targets = []
    all_stress_preds = []
    all_stress_targets = []
    
    model.eval()
    with torch.no_grad():
        for i, (adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types, traj_ids) in enumerate(test_loader):
            gs = [A.to(device) for A in adj_mat_list]
            hs = [X_t.to(device) for X_t in feat_t_mat_list]
            targets = [X_tp1.to(device) for X_tp1 in feat_tp1_mat_list]
            node_types = [nt.to(device) for nt in node_types]
            
            # Forward
            _, preds_list = model(gs, hs, targets, node_types)
            # Remove hook after first batch
            if i == 0:
                handle.remove()

            # Collect per graph
            for pred, target, nt in zip(preds_list, targets, node_types):
                vel_mask = (nt == 0)
                stress_mask = (nt == 0) | (nt == 6)
                
                # Velocity
                if vel_mask.any():
                    p_vel = pred[:, :3][vel_mask]
                    t_vel = target[:, 4:7][vel_mask]
                    all_vel_preds.append(p_vel.cpu().numpy())
                    all_vel_targets.append(t_vel.cpu().numpy())
                    
                # Stress
                if stress_mask.any():
                    p_stress = pred[:, 3:4][stress_mask]
                    t_stress = target[:, 7:8][stress_mask]
                    all_stress_preds.append(p_stress.cpu().numpy())
                    all_stress_targets.append(t_stress.cpu().numpy())

    # Concatenate
    if all_vel_preds:
        cat_vel_preds = np.concatenate(all_vel_preds, axis=0) # [Total_N_vel, 3]
        cat_vel_targets = np.concatenate(all_vel_targets, axis=0)
    else:
        cat_vel_preds = np.zeros((0, 3))
        cat_vel_targets = np.zeros((0, 3))
        
    if all_stress_preds:
        cat_stress_preds = np.concatenate(all_stress_preds, axis=0) # [Total_N_stress, 1]
        cat_stress_targets = np.concatenate(all_stress_targets, axis=0)
    else:
        cat_stress_preds = np.zeros((0, 1))
        cat_stress_targets = np.zeros((0, 1))
        
    # Prepare for plotting: We pass list of arrays: [VelX, VelY, VelZ, Stress]
    final_preds = [cat_vel_preds[:, 0], cat_vel_preds[:, 1], cat_vel_preds[:, 2], cat_stress_preds[:, 0]]
    final_targets = [cat_vel_targets[:, 0], cat_vel_targets[:, 1], cat_vel_targets[:, 2], cat_stress_targets[:, 0]]
    
    make_final_plots(save_dir=PLOTS_DIR, train_losses=train_losses, val_losses=val_losses,
        metric_name='MAE', train_metrics=train_maes, val_metrics=val_maes, grad_norms=grad_norms,
        model=model, activations=activations, predictions=final_preds, targets=final_targets)
    
    print(f"Plots saved to {PLOTS_DIR}")


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
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['lr'], weight_decay=train_cfg['weight_decay'])

    # Training loop
    print("\n=================================================")
    print(" TRAINING")
    print("=================================================\n")
    print(f"Epochs: {train_cfg['epochs']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Learning rate: {train_cfg['lr']}\n")

    # History tracking
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    grad_norms = []

    for epoch in range(train_cfg['epochs']):

        # Train phase
        model.train()
        total_train_loss = 0.0
        total_train_mae = 0.0
        total_train_count = 0
        
        epoch_grad_norm = 0.0
        num_batches = 0

        for adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types, traj_ids in train_loader:
            adj_mat_list = [A.to(device) for A in adj_mat_list]
            feat_t_mat_list = [X_t.to(device) for X_t in feat_t_mat_list]
            feat_tp1_mat_list = [X_tp1.to(device) for X_tp1 in feat_tp1_mat_list]
            node_types = [nt.to(device) for nt in node_types]

            optimizer.zero_grad()
            # One epoch
            batch_loss, preds_list = model(adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, node_types)
            batch_loss.backward()
            
            # Compute grad norm
            norm = get_grad_norm(model)
            epoch_grad_norm += norm
            
            optimizer.step()
            # Accumulate loss
            total_train_loss += batch_loss.item()
            
            # Compute MAE
            mae, count = compute_batch_metrics(preds_list, feat_tp1_mat_list, node_types)
            total_train_mae += mae
            total_train_count += count
            
            num_batches += 1

        # Store epoch average grad norm
        if num_batches > 0:
            grad_norms.append(epoch_grad_norm / num_batches)
            # if gradient clipping, measure after not before, and no .grad parameters are skipepd
        else:
            grad_norms.append(0.0)

        # Evaluation
        model.eval()
        total_test_loss = 0.0
        total_test_mae = 0.0
        total_test_count = 0
        
        with torch.no_grad():
            for adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types, traj_ids in test_loader:
                gs = [A.to(device) for A in adj_mat_list]
                hs = [X_t.to(device) for X_t in feat_t_mat_list]
                targets = [X_tp1.to(device) for X_tp1 in feat_tp1_mat_list]
                node_types = [nt.to(device) for nt in node_types]

                batch_loss, preds_list = model(gs, hs, targets, node_types)
                total_test_loss += batch_loss.item()
                
                mae, count = compute_batch_metrics(preds_list, targets, node_types)
                total_test_mae += mae
                total_test_count += count

        avg_train = total_train_loss / len(train_loader)
        avg_test = total_test_loss / len(test_loader)
        
        avg_train_mae = total_train_mae / total_train_count if total_train_count > 0 else 0.0
        avg_test_mae = total_test_mae / total_test_count if total_test_count > 0 else 0.0
        
        train_losses.append(avg_train)
        val_losses.append(avg_test)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_test_mae)

        print(f"[Epoch {epoch:03d}]  Train Loss: {avg_train:.6f} (MAE: {avg_train_mae:.6f}) |  Test Loss: "
              f"{avg_test:.6f} (MAE: {avg_test_mae:.6f})")

    print(f"\nSaving model to {CHECKPOINT_PATH}")
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    # Final plots
    run_final_evaluation(model, test_loader, device, train_losses, val_losses, train_maes, val_maes, grad_norms)


if __name__ == "__main__":
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_gnet_ema(device)
