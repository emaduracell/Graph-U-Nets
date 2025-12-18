"""
Optuna hyperparameter optimization for Graph-U-Nets (PyTorch) training loop.

This script has NO CLI arguments.
All configuration MUST be specified in `hyperconfig.yaml` in this directory.

It intentionally:
- Does NOT save any model checkpoint
- Does NOT create plots / call final evaluation
- Saves ONLY the best hyperparameter config (YAML) according to validation loss
"""

from __future__ import annotations

import copy
import os
import random
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from torch.amp import autocast

from data.defplate_dataset import DefPlateDataset, collate_unet
from helpers.helpers import (create_model_hyperparams, get_device, get_feature_indices, load_config,
                             load_trajectories_preprocessed, move_any_to_device)
from model_gunet.gunet_deforming_plate import GraphUNet_DefPlate


def _require(cfg, key_path):
    """
    Fetch a required config value using dot-separated path, e.g. 'study.trials'.
    Raises KeyError with a clear message if missing.
    """
    cur: Any = cfg
    parts = key_path.split(".")
    for p in parts:
        if not isinstance(cur, dict) or p not in cur:
            raise KeyError(f"Missing required config key: '{key_path}'")
        cur = cur[p]
    return cur

def _require_list(cfg, key_path):
    v = _require(cfg, key_path)
    if not isinstance(v, list):
        raise TypeError(f"Config key '{key_path}' must be a list")
    return v

def _seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def _compute_single_graph_loss(pred, target, nodetype, velocity_idxs, stress_idxs, normal_node, boundary_node):

    vel_mask = (nodetype == normal_node)
    stress_mask = (nodetype == normal_node) | (nodetype == boundary_node)

    target_vel = target[:, velocity_idxs]
    target_stress = target[:, stress_idxs]
    pred_vel = pred[:, :3]
    pred_stress = pred[:, 3:4]

    vel_loss = torch.zeros((), device=pred.device)
    stress_loss = torch.zeros((), device=pred.device)

    if vel_mask.any():
        vel_loss = F.huber_loss(pred_vel[vel_mask], target_vel[vel_mask])
    if stress_mask.any():
        stress_loss = F.huber_loss(pred_stress[stress_mask], target_stress[stress_mask])

    return vel_loss, stress_loss


def compute_loss(feat_tp1_mat_list, node_types_list, preds_list, velocity_idxs, stress_idxs, normal_node,
                 boundary_node):
    total_loss = torch.zeros((), device=preds_list[0].device)
    total_vel_loss = torch.zeros((), device=preds_list[0].device)
    total_stress_loss = torch.zeros((), device=preds_list[0].device)

    num_graphs = len(preds_list)
    for pred, target, nodetype in zip(preds_list, feat_tp1_mat_list, node_types_list):
        vel_loss, stress_loss = _compute_single_graph_loss(
            pred, target, nodetype, velocity_idxs, stress_idxs, normal_node, boundary_node
        )
        total_vel_loss += vel_loss
        total_stress_loss += stress_loss
        total_loss += vel_loss + stress_loss

    denom = max(num_graphs, 1)
    return total_loss / denom, total_vel_loss / denom, total_stress_loss / denom


@torch.no_grad()
def _validate_one_epoch(model, val_loader, device, velocity_idxs, stress_idxs, amp_enabled, normal_node,
                        boundary_node):

    model.eval()
    total_loss = 0.0
    num_batches = 0

    for batch in tqdm(val_loader, desc="Val", leave=False):
        adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, _, _, _, node_types, _, _ = batch

        adj_mat_list = [A.to(device, non_blocking=True) for A in adj_mat_list]
        feat_t_mat_list = [X.to(device, non_blocking=True) for X in feat_t_mat_list]
        feat_tp1_mat_list = [X.to(device, non_blocking=True) for X in feat_tp1_mat_list]
        node_types = [nt.to(device, non_blocking=True) for nt in node_types]

        if device.type == "cuda":
            with autocast(device_type=device.type, enabled=amp_enabled):
                preds_list = model(adj_mat_list, feat_t_mat_list)
        else:
            preds_list = model(adj_mat_list, feat_t_mat_list)

        batch_loss, _, _ = compute_loss(feat_tp1_mat_list, node_types, preds_list, velocity_idxs,
                                        stress_idxs, normal_node, boundary_node)
        total_loss += float(batch_loss.detach().cpu())
        num_batches += 1

    return total_loss / max(num_batches, 1)


def _train_one_epoch(model, train_loader, optimizer, device, velocity_idxs, stress_idxs,  amp_enabled, scaler,
                     move_all_to_device, normal_node, boundary_node):

    model.train()
    for batch in tqdm(train_loader, desc="Train", leave=False):
        adj_mat_list, feat_t_mat_list, feat_tp1_mat_list, _, _, _, node_types, _, _ = batch

        if (not move_all_to_device) and adj_mat_list[0].device != device:
            adj_mat_list = [A.to(device) for A in adj_mat_list]
            feat_t_mat_list = [X.to(device) for X in feat_t_mat_list]
            feat_tp1_mat_list = [X.to(device) for X in feat_tp1_mat_list]
            node_types = [nt.to(device) for nt in node_types]

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            with autocast(device_type=device.type, enabled=amp_enabled):
                preds_list = model(adj_mat_list, feat_t_mat_list)
                batch_loss, _, _ = compute_loss(feat_tp1_mat_list, node_types, preds_list, velocity_idxs,
                                                stress_idxs, normal_node, boundary_node)
        else:
            preds_list = model(adj_mat_list, feat_t_mat_list)
            batch_loss, _, _ = compute_loss(feat_tp1_mat_list, node_types, preds_list, velocity_idxs, stress_idxs,
                                            normal_node, boundary_node)

        if amp_enabled and scaler is not None:
            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            batch_loss.backward()
            optimizer.step()


def _make_fixed_split_indices(n: int, seed: int, val_fraction: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator()
    g.manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    split = int((1.0 - val_fraction) * n)
    train_idx = perm[:split]
    val_idx = perm[split:]
    return train_idx, val_idx


def _create_dataloaders_for_hpo(dataset, batch_size, num_workers, pin_memory, seed, val_fraction):
    train_idx, val_idx = _make_fixed_split_indices(len(dataset), seed=seed, val_fraction=val_fraction)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    # For fair comparison across trials, keep ordering deterministic (no shuffle).
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, collate_fn=collate_unet,
                              num_workers=num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_unet,
                            num_workers=num_workers, pin_memory=pin_memory)

    return train_loader, val_loader


def _suggest_from_candidates(trial, name, candidates):
    if len(candidates) < 1:
        raise ValueError(f"Candidates list for '{name}' must be non-empty")
    idx = trial.suggest_int(f"{name}_idx", 0, len(candidates) - 1)
    return candidates[int(idx)]


def run_study(hyperconfig_path):
    """
    Run hyperparameter optimization

    Args:
        hyperconfig_path:
    """

    try:
        import optuna  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Optuna is required. Install with: pip install optuna") from e

    hypercfg = load_config(hyperconfig_path)
    seed = int(_require(hypercfg, "study.seed"))
    _seed_all(seed)

    base_config_path = str(_require(hypercfg, "base_config_path"))
    config = load_config(base_config_path)
    model_cfg = config["model"]
    train_cfg = config["training"]

    # Overwrite training meta from hyperconfig (no defaults).
    train_cfg = copy.deepcopy(train_cfg)
    train_cfg["num_train_trajs"] = int(_require(hypercfg, "data.num_train_trajs"))
    train_cfg["mode"] = str(_require(hypercfg, "data.mode"))

    device_cfg = _require(hypercfg, "device")
    if not isinstance(device_cfg, dict):
        raise TypeError("Config key 'device' must be a dict")
    cuda_flag = bool(_require(hypercfg, "device.cuda"))
    device = get_device(cuda_flag)

    # Load dataconfig + indices for features
    dataconfig_path = os.path.join(train_cfg["datapath"], "used_dataconfig.yaml")
    dataconfig = load_config(dataconfig_path)
    include_mesh_pos = dataconfig["include_mesh_pos"]
    feat_idx = get_feature_indices(include_mesh_pos)

    move_all_to_device = bool(train_cfg.get("move_all_to_device"))
    amp_enabled = bool(train_cfg.get("amp"))

    normal_node = int(_require(hypercfg, "constants.normal_node"))
    boundary_node = int(_require(hypercfg, "constants.boundary_node"))
    dim_out_vel = int(_require(hypercfg, "constants.dim_out_vel"))
    dim_out_stress = int(_require(hypercfg, "constants.dim_out_stress"))

    trials = int(_require(hypercfg, "study.trials"))
    epochs = int(_require(hypercfg, "study.epochs"))
    val_fraction = float(_require(hypercfg, "data.val_fraction"))
    out_path = str(_require(hypercfg, "output.best_config_path"))

    batch_size = int(_require(hypercfg, "data.batch_size"))
    num_workers = int(_require(hypercfg, "data.num_workers"))
    pin_memory = bool(_require(hypercfg, "data.pin_memory"))

    # Load trajectories once
    list_of_trajs = load_trajectories_preprocessed(
        os.path.join(train_cfg["datapath"], "preprocessed_train.pt"), train_cfg["num_train_trajs"],
    )
    if move_all_to_device:
        list_of_trajs = move_any_to_device(list_of_trajs, device, non_blocking=False)
        # GPU-resident dataset + multi-worker dataloader is a footgun
        if num_workers != 0:
            print("[hpo] move_all_to_device=True -> forcing num_workers=0")
        num_workers = 0
        pin_memory = False

    dataset = DefPlateDataset(list_of_trajs, world_pos_idxs=feat_idx.world_pos, velocity_idxs=feat_idx.velocity)
    train_loader, val_loader = _create_dataloaders_for_hpo(dataset=dataset, batch_size=batch_size,
                                                           num_workers=num_workers, pin_memory=pin_memory, seed=seed,
                                                           val_fraction=val_fraction)

    depth = len(model_cfg.get("k_pool_ratios"))

    # Search spaces (no defaults; required).
    sp_lr = _require(hypercfg, "search_space.lr")
    sp_wd = _require(hypercfg, "search_space.adam_weight_decay")
    sp_gamma = _require(hypercfg, "search_space.gamma_lr_scheduler")
    sp_dropout = _require_list(hypercfg, "search_space.dropout_candidates")
    sp_activation = _require_list(hypercfg, "search_space.activation_candidates")
    sp_k_pool = _require_list(hypercfg, "search_space.k_pool_ratios_candidates")

    if (not isinstance(sp_lr, dict)) or (not isinstance(sp_wd, dict)) or (not isinstance(sp_gamma, dict)):
        raise TypeError("search_space.lr / adam_weight_decay / gamma_lr_scheduler must be dicts with min/max/log keys")

    def objective(trial: "optuna.trial.Trial") -> float:
        # Keep each trial deterministic given a base seed, but different enough for weights init.
        _seed_all(seed + int(trial.number))

        trial_model_cfg = copy.deepcopy(model_cfg)
        trial_train_cfg = copy.deepcopy(train_cfg)

        trial_train_cfg["lr"] = float(
            trial.suggest_float("lr", float(_require(sp_lr, "min")), float(_require(sp_lr, "max")),
                                log=bool(_require(sp_lr, "log")))
        )
        trial_train_cfg["adam_weight_decay"] = float(
            trial.suggest_float("adam_weight_decay",
                float(_require(sp_wd, "min")),
                float(_require(sp_wd, "max")),
                log=bool(_require(sp_wd, "log")),
            )
        )
        trial_train_cfg["gamma_lr_scheduler"] = float(
            trial.suggest_float(
                "gamma_lr_scheduler",
                float(_require(sp_gamma, "min")),
                float(_require(sp_gamma, "max")),
                log=bool(_require(sp_gamma, "log")),
            )
        )

        dropout = float(trial.suggest_categorical("dropout", [float(x) for x in sp_dropout]))
        trial_model_cfg["dropout_gnn"] = dropout
        trial_model_cfg["dropout_mlps_final"] = dropout

        act = trial.suggest_categorical("activation", [str(x) for x in sp_activation])
        trial_model_cfg["activation_gnn"] = act
        trial_model_cfg["activation_mlps_final"] = act

        # Candidate is a list-of-floats; enforce length matches depth.
        ks = _suggest_from_candidates(trial, "k_pool_ratios", sp_k_pool)
        if (not isinstance(ks, list)) or (len(ks) != depth):
            raise ValueError(f"Each k_pool_ratios candidate must be a list of length {depth}")
        trial_model_cfg["k_pool_ratios"] = [float(v) for v in ks]

        # Build model
        model_hparams = create_model_hyperparams(trial_model_cfg)
        model = GraphUNet_DefPlate(feat_idx.dim_in, dim_out_vel, dim_out_stress, model_hparams, trial_model_cfg["adj_norm"])
        model = model.to(device)

        optimizer = optim.Adam(
            model.parameters(),
            lr=trial_train_cfg["lr"],
            weight_decay=trial_train_cfg["adam_weight_decay"],
            fused=(device.type == "cuda"),
        )
        scheduler = ExponentialLR(optimizer, gamma=trial_train_cfg["gamma_lr_scheduler"])

        from torch.amp import GradScaler
        scaler = GradScaler(enabled=amp_enabled)

        best_val = float("inf")
        for _epoch in range(int(epochs)):
            _train_one_epoch(model=model, train_loader=train_loader, optimizer=optimizer, device=device,
                             velocity_idxs=feat_idx.velocity, stress_idxs=feat_idx.stress, amp_enabled=amp_enabled,
                             scaler=scaler, move_all_to_device=move_all_to_device, normal_node=normal_node,
                             boundary_node=boundary_node)
            val_loss = _validate_one_epoch(model=model, val_loader=val_loader, device=device,
                                           velocity_idxs=feat_idx.velocity, stress_idxs=feat_idx.stress,
                                           amp_enabled=amp_enabled, normal_node=normal_node,
                                           boundary_node=boundary_node)
            best_val = min(best_val, float(val_loss))
            scheduler.step()

            trial.report(best_val, step=_epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Cleanup between trials (important on GPU)
        del model, optimizer, scheduler
        if device.type == "cuda":
            torch.cuda.empty_cache()

        return best_val

    sampler_cfg = _require(hypercfg, "study.sampler")
    pruner_cfg = _require(hypercfg, "study.pruner")
    if not isinstance(sampler_cfg, dict) or not isinstance(pruner_cfg, dict):
        raise TypeError("study.sampler and study.pruner must be dicts")

    sampler_name = str(_require(sampler_cfg, "name")).lower()
    if sampler_name != "tpe":
        raise ValueError("Only sampler.name='tpe' is supported right now (explicitly configured, no defaults).")
    sampler = optuna.samplers.TPESampler(seed=seed)

    pruner_name = str(_require(pruner_cfg, "name")).lower()
    if pruner_name != "median":
        raise ValueError("Only pruner.name='median' is supported right now (explicitly configured, no defaults).")
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(_require(pruner_cfg, "n_startup_trials")),
        n_warmup_steps=int(_require(pruner_cfg, "n_warmup_steps")),
    )
    study = optuna.create_study(direction="minimize", sampler=sampler, pruner=pruner)

    show_progress_bar = bool(_require(hypercfg, "study.show_progress_bar"))
    print(
        f"[hpo] Starting Optuna study: trials={trials}, epochs={epochs}, device={device}, "
        f"num_train_trajs={train_cfg['num_train_trajs']}, val_fraction={val_fraction}"
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=show_progress_bar)

    best = study.best_trial

    # Build best config (model + training only, as requested).
    best_cfg: Dict[str, Any] = {"model": copy.deepcopy(model_cfg), "training": copy.deepcopy(train_cfg)}
    best_cfg["training"]["lr"] = float(best.params["lr"])
    best_cfg["training"]["adam_weight_decay"] = float(best.params["adam_weight_decay"])
    best_cfg["training"]["gamma_lr_scheduler"] = float(best.params["gamma_lr_scheduler"])

    best_cfg["model"]["dropout_gnn"] = float(best.params["dropout"])
    best_cfg["model"]["dropout_mlps_final"] = float(best.params["dropout"])
    best_cfg["model"]["activation_gnn"] = str(best.params["activation"])
    best_cfg["model"]["activation_mlps_final"] = str(best.params["activation"])

    k_idx = int(best.params["k_pool_ratios_idx"])
    best_cfg["model"]["k_pool_ratios"] = [float(v) for v in sp_k_pool[k_idx]]

    # Save + print.
    import yaml

    with open(out_path, "w") as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    print("\n[hpo] Best trial:")
    print(f"  value (best val loss): {best.value}")
    print("  params:")
    for k, v in best.params.items():
        print(f"    - {k}: {v}")
    print(f"\n[hpo] Saved best config to: {out_path}")


def main():
    # All settings must be specified in hyperconfig.yaml (no defaults).
    hyperconfig_path = os.path.join(os.path.dirname(__file__), "hyperconfig.yaml")
    run_study(hyperconfig_path=hyperconfig_path)

if __name__ == "__main__":
    main()


