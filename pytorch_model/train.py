import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
import yaml
import os
import numpy as np
import sys
from defplate_dataset import DefPlateDataset, collate_unet
from model_entire import GraphUNet_DefPlate
from plots import make_final_plots
from torch.optim.lr_scheduler import ExponentialLR
import time

BOUNDARY_NODE = 3
SPHERE_NODE = 1
NORMAL_NODE = 0
DIM_OUT_VEL = 3
DIM_OUT_STRESS = 1

def compute_loss(adj_A_list, feat_tp1_mat_list, node_types_list, preds_list, velocity_idxs, stress_idxs):
    """
    Computes loss per batch

    :param stress_idxs:
    :param velocity_idxs:
    :param adj_A_list:
        batch adjacency matrix list
    :param feat_tp1_mat_list:
        feature matrix list
    :param node_types_list:
        node type list
    :param preds_list:
        list of predictions returned by forward

    :return: (loss_batch_avgd, vel_loss_batch_avgd, stress_loss_batch_avgd)
    """
    # batch_adj_A, batch_feat_X, feat_tp1_mat_list, node_types
    # Loss
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    num_graphs = len(adj_A_list)
    for pred, target, nodetype in zip(preds_list, feat_tp1_mat_list, node_types_list):
        # Create masks for filtering: FOR VEL mask OUT both the sphere and the boundary constraints
        # FOR STRESS mask OUT only the sphere (calculate stress on BC)
        vel_mask = (nodetype == NORMAL_NODE)
        stress_mask = (nodetype == NORMAL_NODE) | (nodetype == BOUNDARY_NODE)
        # Extract targets
        target_vel = target[:, velocity_idxs]
        target_stress = target[:, stress_idxs]
        # Extract predictions
        pred_vel = pred[:, :3]
        pred_stress = pred[:, 3:4]
        # Loss per graph
        loss_graph = 0.0
        # weight_stress = torch.count_nonzero(stress_mask) / (torch.count_nonzero(stress_mask) + torch.count_nonzero(vel_mask))
        weight_stress = 1
        # weight_vel = torch.count_nonzero(vel_mask) / (torch.count_nonzero(stress_mask) + torch.count_nonzero(vel_mask))
        weight_vel = 1
        # print(f"type(vel_mask) = {type(vel_mask)} \t shape(vel_mask) = {vel_mask.shape} \t weight_vel = {weight_vel} "
        #       f"\t {pred_vel[vel_mask].shape}")
        # print(f"type(stress_mask) = {type(stress_mask)} \t shape(stress_mask) = {stress_mask.shape} weight_stress = "
        #       f"\t {weight_stress} \t {pred_vel[stress_mask].shape}")
        # print(f"boundaries = {(nodetype == BOUNDARY_NODE).any()}")

        if vel_mask.any():
            # vel_loss = F.huber_loss(pred_vel[vel_mask], target_vel[vel_mask])
            vel_loss = F.mse_loss(pred_vel[vel_mask], target_vel[vel_mask])
            weighted_vel_loss = vel_loss * weight_vel
            loss_graph = loss_graph + weighted_vel_loss
            total_vel_loss = total_vel_loss + weighted_vel_loss
            # print(f"\t \t Velocity loss = {F.mse_loss(pred_vel[vel_mask], target_vel[vel_mask])}")
        if stress_mask.any():
            # stress_loss = F.mse_loss(pred_stress[stress_mask], target_stress[stress_mask])
            stress_loss = F.huber_loss(pred_stress[stress_mask], target_stress[stress_mask])
            weighted_stress_loss = weight_stress * stress_loss
            loss_graph = loss_graph + weighted_stress_loss
            total_stress_loss = total_stress_loss + weighted_stress_loss
            # print(f"\t \t Stress loss = {F.mse_loss(pred_stress[stress_mask], target_stress[stress_mask])}")
        # Total loss
        total_loss = total_loss + loss_graph

    loss_batch_avgd = total_loss / num_graphs
    vel_loss_batch_avgd = total_vel_loss / num_graphs
    stress_loss_batch_avgd = total_stress_loss / num_graphs
    return loss_batch_avgd, vel_loss_batch_avgd, stress_loss_batch_avgd


def load_config(config_path):
    """Load model and training configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def get_grad_norm(model):
    """
    Get gradient norm of current batch (L2)

    :param model:
        model object

    :return: total norm of current batch
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


def compute_batch_metrics(preds_list, targets_list, node_types_list, velocity_idxs, stress_idxs):
    """Compute MAE for the batch."""
    total_mae = 0.0
    count = 0

    for pred, target, nt in zip(preds_list, targets_list, node_types_list):
        vel_mask = (nt == NORMAL_NODE)
        stress_mask = (nt == NORMAL_NODE) | (nt == BOUNDARY_NODE)

        target_vel = target[:, velocity_idxs]
        target_stress = target[:, stress_idxs]
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


def run_final_evaluation(model, test_loader, device, train_losses, val_losses, train_maes, val_maes, grad_norms,
                         train_vel_losses, train_stress_losses, test_vel_losses, test_stress_losses, velocity_idxs,
                         stress_idxs, plots_dir):
    """
    Runs evaluation, collecting BOTH Normalized and Denormalized data for plotting.
    """
    print("[train] Generating final evaluation plots...")

    # activations (using a hook on one batch)
    activations = {}
    handle = model.velocity_mlp.register_forward_hook(
        lambda m, i, o: activations.update({'latent_features': i[0].detach().cpu().numpy()})
    )

    # Containers for Denormalized Data (Physical Units)
    all_vel_preds = []
    all_vel_targets = []
    all_stress_preds = []
    all_stress_targets = []

    # Containers for Normalized Data (Model Output Scale)
    all_vel_preds_norm = []
    all_vel_targets_norm = []
    all_stress_preds_norm = []
    all_stress_targets_norm = []

    model.eval()
    with torch.no_grad():
        for i, (adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types, dyn_edges,
                traj_ids, time_ids, batch_compute_times) in enumerate(test_loader):

            print(f"[run_final_evaluation] batch {i}")
            gs = [A.to(device) for A in adj_mat_list]
            hs = [X_t.to(device) for X_t in feat_t_mat_list]
            targets = [X_tp1.to(device) for X_tp1 in feat_tp1_mat_list]
            node_types_gpu = [nt.to(device) for nt in node_types]

            # Forward pass (Normalized domain)
            preds_list = model(gs, hs, targets, node_types_gpu)

            if i == 0:
                handle.remove()

            # Iterate over graphs in the batch
            for pred, target, nodetype, mean, std in zip(preds_list, targets, node_types_gpu, means, stds):

                # 1. Prepare Data
                # Squeeze to shape [Features] to make indexing easy
                mean = mean.to(device).squeeze()
                std = std.to(device).squeeze()

                # --- FIX FOR PRED DENORMALIZATION ---
                # We need mean/std vectors that exactly match pred shape [3 vel, 1 stress]

                # Extract velocity stats
                std_vel = std[velocity_idxs]  # Shape [3]
                mean_vel = mean[velocity_idxs]  # Shape [3]

                # Extract stress stats
                std_stress = std[stress_idxs]  # Shape [1]
                mean_stress = mean[stress_idxs]  # Shape [1]

                # Concatenate to make shape [4] matching 'pred'
                std_pred_subset = torch.cat([std_vel, std_stress])
                mean_pred_subset = torch.cat([mean_vel, mean_stress])

                # Now shapes match: [N, 4] * [4] + [4]
                pred_denorm = pred * std_pred_subset + mean_pred_subset

                # --- FIX FOR TARGET DENORMALIZATION ---
                # Target has ALL features, so we can use the FULL mean/std
                # Shape: [N, F] * [F] + [F]
                target_denorm = target * std + mean

                # 2. Strict Masking (Only evaluate NORMAL nodes)
                eval_mask = (nodetype == NORMAL_NODE)

                if eval_mask.any():
                    # --- Collect Denormalized ---
                    # Now we can slice safely.
                    # pred_denorm is [N, 4] -> use integers 0:3 and 3:4
                    p_vel = pred_denorm[:, :3][eval_mask]
                    p_stress = pred_denorm[:, 3:4][eval_mask]

                    # target_denorm is [N, F] -> use the slice objects (e.g. 8:11)
                    t_vel = target_denorm[:, velocity_idxs][eval_mask]
                    t_stress = target_denorm[:, stress_idxs][eval_mask]

                    all_vel_preds.append(p_vel.cpu().numpy())
                    all_vel_targets.append(t_vel.cpu().numpy())
                    all_stress_preds.append(p_stress.cpu().numpy())
                    all_stress_targets.append(t_stress.cpu().numpy())

                    # --- Collect Normalized ---
                    p_vel_norm = pred[:, :3][eval_mask]
                    t_vel_norm = target[:, velocity_idxs][eval_mask]
                    p_stress_norm = pred[:, 3:4][eval_mask]
                    t_stress_norm = target[:, stress_idxs][eval_mask]

                    all_vel_preds_norm.append(p_vel_norm.cpu().numpy())
                    all_vel_targets_norm.append(t_vel_norm.cpu().numpy())
                    all_stress_preds_norm.append(p_stress_norm.cpu().numpy())
                    all_stress_targets_norm.append(t_stress_norm.cpu().numpy())

    # --- Helper to Concatenate Lists ---
    def concat_or_empty(preds_list, targets_list, dim):
        if preds_list:
            p = np.concatenate(preds_list, axis=0)
            t = np.concatenate(targets_list, axis=0)
            return p, t
        else:
            return np.zeros((0, dim)), np.zeros((0, dim))

    # Concatenate Denormalized
    cat_vel_preds, cat_vel_targets = concat_or_empty(all_vel_preds, all_vel_targets, 3)
    cat_stress_preds, cat_stress_targets = concat_or_empty(all_stress_preds, all_stress_targets, 1)

    # Concatenate Normalized
    cat_vel_preds_norm, cat_vel_targets_norm = concat_or_empty(all_vel_preds_norm, all_vel_targets_norm, 3)
    cat_stress_preds_norm, cat_stress_targets_norm = concat_or_empty(all_stress_preds_norm, all_stress_targets_norm, 1)

    # Prepare lists for plotting [VelX, VelY, VelZ, Stress]

    # 1. Denormalized Lists
    final_preds = [cat_vel_preds[:, 0], cat_vel_preds[:, 1], cat_vel_preds[:, 2], cat_stress_preds[:, 0]]
    final_targets = [cat_vel_targets[:, 0], cat_vel_targets[:, 1], cat_vel_targets[:, 2], cat_stress_targets[:, 0]]

    # 2. Normalized Lists
    final_preds_norm = [cat_vel_preds_norm[:, 0], cat_vel_preds_norm[:, 1], cat_vel_preds_norm[:, 2],
                        cat_stress_preds_norm[:, 0]]
    final_targets_norm = [cat_vel_targets_norm[:, 0], cat_vel_targets_norm[:, 1], cat_vel_targets_norm[:, 2],
                          cat_stress_targets_norm[:, 0]]

    make_final_plots(save_dir=plots_dir, train_losses=train_losses, val_losses=val_losses,
                     metric_name='MAE', train_metrics=train_maes, val_metrics=val_maes, grad_norms=grad_norms,
                     model=model, activations=activations,

                     # Pass BOTH sets of data
                     predictions=final_preds, targets=final_targets,
                     predictions_norm=final_preds_norm, targets_norm=final_targets_norm,

                     train_vel_losses=train_vel_losses, train_stress_losses=train_stress_losses,
                     test_vel_losses=test_vel_losses, test_stress_losses=test_stress_losses,
                     velocity_idxs=velocity_idxs, stress_idxs=stress_idxs)

    print(f"Plots saved to {plots_dir}")


def train_gnet(device, num_workers, pin_memory):
    """Training loop"""
    # Load configuration from YAML
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)
    # Extract model and training parameters
    model_cfg = config['model']
    train_cfg = config['training']
    # Load train config
    mode = train_cfg['mode']
    start_lr = train_cfg['lr']
    gamma_lr_scheduler = train_cfg['gamma_lr_scheduler']
    adam_weight_decay = train_cfg['adam_weight_decay']
    num_train_trajs = train_cfg['num_train_trajs']
    batch_size = train_cfg['batch_size']
    shuffle = train_cfg['shuffle']
    add_world_edges = train_cfg['add_world_edges']
    preprocessed_data_path = train_cfg['datapath']
    checkpoint_path = (train_cfg['model_path'] + "model_" + preprocessed_data_path.rsplit("/", 1)[0] + "_" +
                       add_world_edges)
    plots_dir = os.path.join(os.path.dirname(__file__), train_cfg['model_path'] + "plots_" +
                             preprocessed_data_path.rsplit("/", 1)[0] + "_" + add_world_edges)
    random_seed = train_cfg['random_seed']
    radius = train_cfg['radius_world_edge']
    k_neighb = train_cfg['k_neighb']
    if "False" in preprocessed_data_path:
        include_mesh_pos = False
    else:
        include_mesh_pos = True

    if include_mesh_pos:
        world_pos_idxs = slice(3, 6)
        velocity_idxs = slice(8, 11)
        stress_idxs = slice(11, 12)
        dim_in = 12 + 3  # mesh_pos (3) + world_pos (3) + node_type (2) + vel (3) + stress (1) + kinematic_vel_tp1 (3)
    else:
        world_pos_idxs = slice(0, 3)
        velocity_idxs = slice(5, 8)
        stress_idxs = slice(8, 9)
        dim_in = 9 + 3 # world_pos (3) + node_type (2) + vel (3) + stress (1) + kinematic_vel_tp1 (3)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
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
    print(" LOADING PREPROCESSED DATA")
    print("=================================================\n")
    print(f"\t Preprocessed data: {preprocessed_data_path}")

    # Load preprocessed trajectories
    if not os.path.exists(preprocessed_data_path):
        raise FileNotFoundError(
            f"Preprocessed data not found at {preprocessed_data_path}\n"
            f"Please run 'python preprocess_data.py' first to generate the preprocessed data."
        )

    list_of_trajs = torch.load(preprocessed_data_path)
    print(f"\t Loaded {len(list_of_trajs)} preprocessed trajectories")

    # Limit to num_train_trajs if specified
    if num_train_trajs is not None and num_train_trajs < len(list_of_trajs):
        list_of_trajs = list_of_trajs[:num_train_trajs]
        print(f"\t Using first {num_train_trajs} trajectories")

    # TODO SPECIFY WHICH TIME STEPS AND TRAJECTORIES WE'RE TRAINING ONE AND WHICH ONES ARE IN THE TEST SET INSTEAD
    # TODO FOR 80-20 SPLIT
    # Build dataset from these trajectories
    dataset = DefPlateDataset(list_of_trajs, add_world_edges=add_world_edges, k_neighb=k_neighb, radius=radius,
                              world_pos_idxs=world_pos_idxs, velocity_idxs=velocity_idxs)
    print(f"Total training pairs (X_t, X_t+1): {len(dataset)}")
    # Random 80/20 split and then load data
    total = len(dataset)
    perm = torch.randperm(total)
    split = int(0.8 * total)
    train_idx = perm[:split]
    test_idx = perm[split:]
    train_set = Subset(dataset, train_idx)
    test_set = Subset(dataset, test_idx)
    # Create a data loader that respects batch size. One batch := a set of graphs, each corresponding to a couple
    # (traj_id, time_id), such that number_of_graphs = batch_size
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_unet,
                              num_workers=num_workers, pin_memory=pin_memory)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, collate_fn=collate_unet,
                             num_workers=num_workers, pin_memory=pin_memory)
    # train_loader = DataLoader(..., drop_last=True)
    # test_loader = DataLoader(..., drop_last=False)

    if mode == "overfit":
        # Get overfit configuration
        overfit_traj_id = train_cfg.get('overfit_traj_id', None)
        overfit_time_idx_list = train_cfg.get('overfit_time_idx', None)

        # Filter dataset to match overfit criteria
        overfit_indices = []
        print(f"len(dataset)={len(dataset)}")
        for idx in range(len(dataset)):
            for i in overfit_time_idx_list:
                sample = dataset.samples[idx]
                # Match trajectory
                if overfit_traj_id is not None and sample['traj_id'] != overfit_traj_id:
                    continue
                # Match time step
                if overfit_time_idx_list is not None and sample['time_idx'] != i:
                    continue
                overfit_indices.append(idx)

        if len(overfit_indices) == 0:
            raise ValueError(f"No samples found matching overfit criteria: traj_id={overfit_traj_id}, "
                             f"time_idx={overfit_time_idx_list}")

        # Create overfit subset
        overfit_set = Subset(dataset, overfit_indices)
        train_loader = DataLoader(overfit_set, batch_size=len(overfit_indices), shuffle=False, collate_fn=collate_unet)
        test_loader = train_loader

        print(f"\nOverfitting on trajectory {overfit_traj_id} with {len(overfit_indices)} time steps")
        overfit_batch = next(iter(train_loader))
        (adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types_cpu, dyn_edges, traj_ids,
         time_indices, batch_compute_times) = overfit_batch
        print("Overfitting on the following (traj_id, time_idx) pairs:")
        for i, (tr, ti) in enumerate(zip(traj_ids, time_indices)):
            print(f"  sample {i:02d}: traj_id={int(tr)}, t={int(ti)}")

    # Build model
    # dim_in = list_of_trajs[0]["X_seq_norm"].shape[2]
    model = GraphUNet_DefPlate(dim_in, DIM_OUT_VEL, DIM_OUT_STRESS, model_hyperparams).to(device)
    optimizer = optim.Adam(model.parameters(), lr=start_lr, weight_decay=adam_weight_decay)
    scheduler = ExponentialLR(optimizer, gamma=gamma_lr_scheduler)

    # Training loop
    print("\n=================================================")
    print(" \t \t TRAINING")
    print("=================================================\n")
    print(f"Epochs: {train_cfg['epochs']}")
    print(f"Batch size: {train_cfg['batch_size']}")
    print(f"Start learning rate: {start_lr}\n")
    print(f"Mode: {mode}")
    print(f"Weight decay = {adam_weight_decay}")
    print(f"Number of trajectories on which I'm training = {num_train_trajs}")
    print(f"len(train_loader) = {len(train_loader)}")

    # History tracking
    train_losses = []
    val_losses = []
    train_maes = []
    train_vel_losses = []
    train_stress_losses = []
    test_vel_losses = []
    test_stress_losses = []
    val_maes = []
    grad_norms = []
    start_time_training = time.time()
    global_total_edge_time = 0.0
    global_total_edge_calls = 0

    for epoch in range(train_cfg['epochs']):

        # Train phase
        model.train()
        total_train_loss = 0.0
        total_train_vel_loss = 0.0
        total_train_stress_loss = 0.0
        total_train_mae = 0.0
        total_train_count = 0

        epoch_grad_norm = 0.0
        num_batches = 0

        for adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types_cpu, dyn_edges, traj_ids \
                , time_ids, batch_compute_times, in train_loader:

            global_total_edge_time += sum(batch_compute_times)
            global_total_edge_calls += len(batch_compute_times)

            adj_mat_list = [A.to(device) for A in adj_mat_list]
            feat_t_mat_list = [X_t.to(device) for X_t in feat_t_mat_list]
            feat_tp1_mat_list = [X_tp1.to(device) for X_tp1 in feat_tp1_mat_list]
            node_types_cpu = [nt.to(device) for nt in node_types_cpu]

            optimizer.zero_grad()
            # One batch: forward and backprop
            # batch_loss, preds_list = model(adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, node_types)
            preds_list = model(adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, node_types_cpu)
            batch_loss, vel_loss_batch_avgd, stress_loss_batch_avgd = compute_loss(adj_mat_list, feat_tp1_mat_list,
                                                                                   node_types_cpu, preds_list,
                                                                                   velocity_idxs, stress_idxs)
            batch_loss.backward()
            # Compute grad norm
            norm = get_grad_norm(model)
            epoch_grad_norm += norm
            # optimizer step
            optimizer.step()
            # Accumulate loss (item for extracting float from 0-dim tensor)
            total_train_loss += batch_loss.item()
            total_train_vel_loss += vel_loss_batch_avgd.item()
            total_train_stress_loss += stress_loss_batch_avgd.item()
            # Compute MAE
            mae, count = compute_batch_metrics(preds_list, feat_tp1_mat_list, node_types_cpu, velocity_idxs,
                                               stress_idxs)
            total_train_mae += mae
            total_train_count += count
            # batches count
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
        total_test_stress_loss = 0.0
        total_test_vel_loss = 0.0
        total_test_mae = 0.0
        total_test_count = 0

        with torch.no_grad():
            for adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, means, stds, cells, node_types, dynamic_edges, \
                traj_ids, time_ids, _ in test_loader:
                adj_mat_cpu_list = [A.to(device) for A in adj_mat_list]
                feat_mat_cpu_list = [X_t.to(device) for X_t in feat_t_mat_list]
                feat_mat_cpu_list_tp1 = [X_tp1.to(device) for X_tp1 in feat_tp1_mat_list]
                node_types_cpu = [nt.to(device) for nt in node_types]

                preds_list = model(adj_mat_cpu_list, feat_mat_cpu_list, feat_mat_cpu_list_tp1, node_types_cpu)
                batch_loss, vel_loss_batch_avgd, stress_loss_batch_avgd = \
                    compute_loss(adj_mat_cpu_list, feat_mat_cpu_list_tp1, node_types_cpu, preds_list, velocity_idxs,
                                 stress_idxs)

                total_test_vel_loss += vel_loss_batch_avgd.item()
                total_test_stress_loss += stress_loss_batch_avgd.item()
                total_test_loss += batch_loss.item()

                mae, count = compute_batch_metrics(preds_list, feat_mat_cpu_list_tp1, node_types_cpu, velocity_idxs,
                                                   stress_idxs)
                total_test_mae += mae
                total_test_count += count

        avg_train = total_train_loss / len(train_loader)
        avg_test = total_test_loss / len(test_loader)
        avg_train_stress_loss = total_train_stress_loss / len(train_loader)
        avg_train_vel_loss = total_train_vel_loss / len(train_loader)
        avg_test_stress_loss = total_test_stress_loss / len(test_loader)
        avg_test_vel_loss = total_test_vel_loss / len(test_loader)

        avg_train_mae = total_train_mae / total_train_count if total_train_count > 0 else 0.0
        avg_test_mae = total_test_mae / total_test_count if total_test_count > 0 else 0.0

        train_losses.append(avg_train)
        train_vel_losses.append(avg_train_vel_loss)
        train_stress_losses.append(avg_train_stress_loss)
        test_vel_losses.append(avg_test_vel_loss)
        test_stress_losses.append(avg_test_stress_loss)
        val_losses.append(avg_test)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_test_mae)

        # lr scheduler
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"[Train] [Epoch = {epoch:03d}]  Train Loss: {avg_train:.6f} (MAE: {avg_train_mae:.6f}) |  Test Loss: "
              f"{avg_test:.6f} (MAE: {avg_test_mae:.6f}) | Velocity loss = "
              f"{avg_train_vel_loss:.6f} | Stress loss = {avg_train_stress_loss:.6f} | Lr = {current_lr:.6f} ")

    total_time_training = time.time() - start_time_training
    hours = int(total_time_training // 3600)
    minutes = int((total_time_training % 3600) // 60); seconds = int(total_time_training % 60);
    time_str = f"[train] Total training time: {hours}h {minutes}m {seconds}s"; print(f"\n{time_str}")

    if global_total_edge_calls > 0:
        avg_edge_time = global_total_edge_time / global_total_edge_calls
        print(f"\n[train] Average 'add_edges' computation time: {avg_edge_time:.6f} seconds per graph")
        print(f"[train] Total edge computation time accumulated: {global_total_edge_time:.2f} seconds")
    else:
        print("[train] No edges added.")

    print(f"\n[train] Saving model to {checkpoint_path}")
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
    torch.save(model.state_dict(), checkpoint_path)

    # Final plots
    run_final_evaluation(model, test_loader, device, train_losses, val_losses, train_maes, val_maes, grad_norms,
                         train_vel_losses, train_stress_losses, test_vel_losses, test_stress_losses, velocity_idxs,
                         stress_idxs, plots_dir)


if __name__ == "__main__":
    num_workers = 8
    pin_memory = True
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    train_gnet(device, num_workers, pin_memory)
