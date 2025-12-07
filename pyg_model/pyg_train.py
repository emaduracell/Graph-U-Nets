import os
import yaml
import torch
from torch_geometric.loader import DataLoader

import torch.nn.functional as F

from pyg_graph_unet_data import GraphUNetTFRecordDataset
from pyg_graph_unet_model import GraphUNetDefPlatePyG
from pyg_plots import make_final_plots


BOUNDARY_NODE = 3
NORMAL_NODE = 0
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "plots")


def load_config(config_path: str):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_device(name: str):
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
    return torch.device(name)


def compute_loss(preds, targets, node_type):
    """MSE with masks: velocity on normals, stress on normals+boundary."""
    vel_mask = (node_type == NORMAL_NODE)
    stress_mask = (node_type == NORMAL_NODE) | (node_type == BOUNDARY_NODE)

    pred_vel = preds[:, :3]
    pred_stress = preds[:, 3:4]
    target_vel = targets[:, :3]
    target_stress = targets[:, 3:4]

    vel_count = vel_mask.sum().item()
    stress_count = stress_mask.sum().item()
    total = max(vel_count + stress_count, 1)
    weight_vel = vel_count / total
    weight_stress = stress_count / total

    loss = 0.0
    vel_loss = torch.tensor(0.0, device=preds.device)
    stress_loss = torch.tensor(0.0, device=preds.device)

    if vel_count > 0:
        vel_loss = F.mse_loss(pred_vel[vel_mask], target_vel[vel_mask])
        loss = loss + weight_vel * vel_loss
    if stress_count > 0:
        stress_loss = F.mse_loss(pred_stress[stress_mask], target_stress[stress_mask])
        loss = loss + weight_stress * stress_loss

    return loss, vel_loss, stress_loss


def compute_mae(preds, targets, node_type):
    vel_mask = (node_type == NORMAL_NODE)
    stress_mask = (node_type == NORMAL_NODE) | (node_type == BOUNDARY_NODE)

    mae_sum = 0.0
    count = 0
    if vel_mask.any():
        mae_sum += F.l1_loss(preds[:, :3][vel_mask], targets[:, :3][vel_mask], reduction="sum").item()
        count += vel_mask.sum().item() * 3
    if stress_mask.any():
        mae_sum += F.l1_loss(preds[:, 3:4][stress_mask], targets[:, 3:4][stress_mask], reduction="sum").item()
        count += stress_mask.sum().item()
    return mae_sum, count


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    total_mae = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        preds = model(batch.x, batch.edge_index, batch=batch.batch if hasattr(batch, "batch") else None)
        loss, vel_loss, stress_loss = compute_loss(preds, batch.y, batch.node_type)
        loss.backward()
        optimizer.step()

        mae_sum, count = compute_mae(preds, batch.y, batch.node_type)
        total_loss += loss.item()
        total_vel_loss += vel_loss.item()
        total_stress_loss += stress_loss.item()
        total_mae += mae_sum
        total_count += count

    num_batches = max(len(loader), 1)
    avg_loss = total_loss / num_batches
    avg_vel_loss = total_vel_loss / num_batches
    avg_stress_loss = total_stress_loss / num_batches
    avg_mae = total_mae / total_count if total_count > 0 else 0.0
    return avg_loss, avg_vel_loss, avg_stress_loss, avg_mae


@torch.no_grad()
def eval_epoch(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_vel_loss = 0.0
    total_stress_loss = 0.0
    total_mae = 0.0
    total_count = 0

    for batch in loader:
        batch = batch.to(device)
        preds = model(batch.x, batch.edge_index, batch=batch.batch if hasattr(batch, "batch") else None)
        loss, vel_loss, stress_loss = compute_loss(preds, batch.y, batch.node_type)

        mae_sum, count = compute_mae(preds, batch.y, batch.node_type)
        total_loss += loss.item()
        total_vel_loss += vel_loss.item()
        total_stress_loss += stress_loss.item()
        total_mae += mae_sum
        total_count += count

    num_batches = max(len(loader), 1)
    avg_loss = total_loss / num_batches
    avg_vel_loss = total_vel_loss / num_batches
    avg_stress_loss = total_stress_loss / num_batches
    avg_mae = total_mae / total_count if total_count > 0 else 0.0
    return avg_loss, avg_vel_loss, avg_stress_loss, avg_mae


def main():
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    config = load_config(config_path)

    data_cfg = config["data"]
    model_cfg = config["model"]
    train_cfg = config["training"]

    device = get_device(train_cfg.get("device", "auto"))
    torch.manual_seed(train_cfg.get("seed", 42))

    # Dataset with optional trajectory/time filtering
    dataset = GraphUNetTFRecordDataset(
        data_dir=os.path.join(os.path.dirname(__file__), data_cfg["data_dir"]),
        split=data_cfg.get("split", "train"),
        preprocessed_dir=os.path.join(
            os.path.dirname(__file__),
            data_cfg.get("preprocessed_dir", "data/graph_unet_preprocessed"),
        ),
        allowed_traj_ids=data_cfg.get("selected_traj_ids"),
        allowed_time_ids=data_cfg.get("selected_time_ids"),
    )

    mode = train_cfg.get("mode", "standard")

    if mode == "overfit":
        overfit_traj = train_cfg.get("overfit_traj_id")
        overfit_time_idx = train_cfg.get("overfit_time_idx")
        if overfit_traj is None:
            raise ValueError("overfit_traj_id must be set when mode == 'overfit'")

        # Narrow dataset to the overfit specification
        dataset = GraphUNetTFRecordDataset(
            data_dir=os.path.join(os.path.dirname(__file__), data_cfg["data_dir"]),
            split=data_cfg.get("split", "train"),
            preprocessed_dir=os.path.join(
                os.path.dirname(__file__),
                data_cfg.get("preprocessed_dir", "data/graph_unet_preprocessed"),
            ),
            allowed_traj_ids=[overfit_traj],
            allowed_time_ids=overfit_time_idx,
        )
        # Single loader used for both train/val to mimic overfit behavior
        train_loader = DataLoader(
            dataset,
            batch_size=len(dataset),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
        )
        val_loader = train_loader
        print(f"Overfitting on traj_id={overfit_traj} with {len(dataset)} samples"
              f"{'' if not overfit_time_idx else f' at times {overfit_time_idx}'}")
    else:
        total_len = len(dataset)
        val_split = train_cfg.get("val_split", 0.2)
        split_idx = int((1 - val_split) * total_len)
        generator = torch.Generator().manual_seed(train_cfg.get("seed", 42))
        perm = torch.randperm(total_len, generator=generator)
        train_idx = perm[:split_idx]
        val_idx = perm[split_idx:]

        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)

        train_loader = DataLoader(
            train_set,
            batch_size=train_cfg.get("batch_size", 2),
            shuffle=train_cfg.get("shuffle", True),
            num_workers=train_cfg.get("num_workers", 0),
        )
        val_loader = DataLoader(
            val_set,
            batch_size=train_cfg.get("batch_size", 2),
            shuffle=False,
            num_workers=train_cfg.get("num_workers", 0),
        )

    model = GraphUNetDefPlatePyG(
        in_channels=dataset.feature_dim,
        hidden_channels=model_cfg["hidden_channels"],
        depth=model_cfg["depth"],
        pool_ratios=model_cfg["pool_ratios"],
        mlp_hidden=model_cfg["mlp_hidden"],
        mlp_dropout=model_cfg.get("mlp_dropout", 0.0),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=train_cfg.get("lr", 1e-3),
        weight_decay=train_cfg.get("weight_decay", 0.0),
    )

    checkpoint_path = os.path.join(os.path.dirname(__file__), train_cfg.get("checkpoint_path", "pyg_graph_unet.pt"))
    epochs = train_cfg.get("epochs", 50)

    # tracking
    train_losses, val_losses = [], []
    train_vel_losses, train_stress_losses = [], []
    val_vel_losses, val_stress_losses = [], []

    for epoch in range(epochs):
        tr_loss, tr_vel, tr_str, tr_mae = train_epoch(model, train_loader, optimizer, device)
        val_loss, val_vel, val_str, val_mae = eval_epoch(model, val_loader, device)

        train_losses.append(tr_loss)
        val_losses.append(val_loss)
        train_vel_losses.append(tr_vel)
        train_stress_losses.append(tr_str)
        val_vel_losses.append(val_vel)
        val_stress_losses.append(val_str)

        print(
            f"[Epoch {epoch:03d}] "
            f"Train Loss {tr_loss:.6f} (vel {tr_vel:.6f}, stress {tr_str:.6f}, MAE {tr_mae:.6f}) | "
            f"Val Loss {val_loss:.6f} (vel {val_vel:.6f}, stress {val_str:.6f}, MAE {val_mae:.6f})"
        )

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "config": config,
                "mean_feat": dataset.mean_feat,
                "std_feat": dataset.std_feat,
                "mean_target": dataset.mean_target,
                "std_target": dataset.std_target,
            },
            checkpoint_path,
        )

    # Final predictions on validation for plotting
    model.eval()
    all_preds = []
    all_tgts = []
    with torch.no_grad():
        for batch in val_loader:
            batch = batch.to(device)
            preds = model(batch.x, batch.edge_index, batch=batch.batch if hasattr(batch, "batch") else None)
            # denormalize velocity+stress targets/preds
            preds_denorm = preds * dataset.std_target.to(device) + dataset.mean_target.to(device)
            tgts_denorm = batch.y * dataset.std_target.to(device) + dataset.mean_target.to(device)
            # apply masks consistent with loss
            vel_mask = (batch.node_type == NORMAL_NODE)
            stress_mask = (batch.node_type == NORMAL_NODE) | (batch.node_type == BOUNDARY_NODE)

            # velocity
            if vel_mask.any():
                all_preds.append(preds_denorm[vel_mask][:, :3].cpu())
                all_tgts.append(tgts_denorm[vel_mask][:, :3].cpu())
            # stress
            if stress_mask.any():
                all_preds.append(preds_denorm[stress_mask][:, 3:4].cpu())
                all_tgts.append(tgts_denorm[stress_mask][:, 3:4].cpu())

    if all_preds and all_tgts:
        preds_cat = torch.cat(all_preds, dim=0)
        tgts_cat = torch.cat(all_tgts, dim=0)
        make_final_plots(
            save_dir=PLOTS_DIR,
            train_losses=train_losses,
            val_losses=val_losses,
            train_vel_losses=train_vel_losses,
            train_stress_losses=train_stress_losses,
            val_vel_losses=val_vel_losses,
            val_stress_losses=val_stress_losses,
            predictions=preds_cat.numpy(),
            targets=tgts_cat.numpy(),
        )


if __name__ == "__main__":
    main()


