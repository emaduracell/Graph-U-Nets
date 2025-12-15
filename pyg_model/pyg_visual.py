import torch
import numpy as np
import yaml
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pyg_model import GraphUNetDefPlatePyG
from torch_geometric.utils import to_dense_adj

# Data paths
# Adjust these paths to match your directory structure
PREPROCESSED_DIR = "/scratch/izar/durante/graph_unets/graph_unet_preprocessed" # Directory containing traj files
STATS_PATH = os.path.join(PREPROCESSED_DIR, "graph_unet_stats.pt")
CHECKPOINT_PATH = "pyg_graph_unet.pt" # Path to your trained model checkpoint
CONFIG_PATH = "config.yaml"

# Constants matches your new setup
BOUNDARY_NODE = 3
NORMAL_NODE = 0
SPHERE_NODE = 1

# Feature Slices (Matches your new 12-dim input)
# [MeshPos(3), WorldPos(3), NodeType(2), Vel(3), Stress(1)]
IDX_MESH_POS = slice(0, 3)
IDX_WORLD_POS = slice(3, 6)
IDX_NODE_TYPE = slice(6, 8)
IDX_VELOCITY = slice(8, 11)
IDX_STRESS = slice(11, 12)

# Visualization settings
TRAJ_INDEX = 0
T_STEP = 0 # time index t (visualize t -> t+1)
ROLLOUT = True  # if True, run multi-step rollout
ROLLOUT_STEPS = 50  # maximum number of rollout steps for multi-step visualization
RENDER_MODE = "all"  # options: "all", "no_border", "no_sphere", "no_border_no_sphere"


def make_wireframe(x, y, z, i, j, k, color='black', width=1.5):
    """
    Creates a Scatter3d trace that draws the edges of the triangles.
    """
    tri_points = np.vstack([
        i, j, k, i, 
        np.full_like(i, -1) 
    ]).T.flatten()
    
    xe = x[tri_points]
    ye = y[tri_points]
    ze = z[tri_points]
    
    xe[4::5] = None
    ye[4::5] = None
    ze[4::5] = None

    return go.Scatter3d(
        x=xe, y=ye, z=ze,
        mode='lines',
        line=dict(color=color, width=width),
        name='wireframe',
        showlegend=False,
        hoverinfo='skip' 
    )

def load_config(config_path):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def visualize_mesh_pair(pos_true, pos_pred, cells, stress_true, stress_pred, node_type_true, node_type_pred, title_true,
                        title_pred, color_mode):
    # (Same as your provided code - purely visualization logic)
    # ... [Keep this function exactly as is] ...
    
    # 1. Remove Batch Dim
    if pos_true.ndim == 3: raise ValueError("pos_true should not have a batch dimension")
    if pos_pred.ndim == 3: raise ValueError("pos_pred should not have a batch dimension")

    # 2. Triangulation
    tri_i, tri_j, tri_k = [], [], []
    for (i0, i1, i2, i3) in cells:
        tri_i.extend([i0, i0, i0, i1])
        tri_j.extend([i1, i2, i3, i3])
        tri_k.extend([i2, i3, i1, i2])

    # 3. Colors
    if color_mode == "stress":
        intensity_true = stress_true.astype(float) if stress_true is not None else np.zeros(pos_true.shape[0])
        intensity_pred = stress_pred.astype(float) if stress_pred is not None else np.zeros(pos_pred.shape[0])
        colorscale = "Viridis"
    elif color_mode == "node_type":
        intensity_true = node_type_true.astype(float) if node_type_true is not None else np.zeros(pos_true.shape[0])
        intensity_pred = node_type_pred.astype(float) if node_type_pred is not None else np.zeros(pos_pred.shape[0])
        colorscale = "Turbo"
    else:
        raise ValueError("color_mode must be 'stress' or 'node_type'")

    # 4. Figure
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "scene"}, {"type": "scene"}]], subplot_titles=(title_true, title_pred))

    # True Mesh
    fig.add_trace(go.Mesh3d(x=pos_true[:,0], y=pos_true[:,1], z=pos_true[:,2], i=tri_i, j=tri_j, k=tri_k, intensity=intensity_true, colorscale=colorscale, showscale=True, flatshading=True, opacity=0.85, name="true"), row=1, col=1)
    fig.add_trace(make_wireframe(pos_true[:,0], pos_true[:,1], pos_true[:,2], np.array(tri_i), np.array(tri_j), np.array(tri_k)), row=1, col=1)

    # Pred Mesh
    fig.add_trace(go.Mesh3d(x=pos_pred[:,0], y=pos_pred[:,1], z=pos_pred[:,2], i=tri_i, j=tri_j, k=tri_k, intensity=intensity_pred, colorscale=colorscale, showscale=True, flatshading=True, opacity=0.85, name="pred"), row=1, col=2)
    fig.add_trace(make_wireframe(pos_pred[:,0], pos_pred[:,1], pos_pred[:,2], np.array(tri_i), np.array(tri_j), np.array(tri_k)), row=1, col=2)

    fig.update_scenes(aspectmode="data")
    fig.update_layout(height=600, width=1200, title_text="Mesh Comparison")
    fig.show()

def apply_render_mode(pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells):
    # (Same as your provided code)
    # ... [Keep this function exactly as is] ...
    mode = RENDER_MODE.lower()
    if mode == "all": return pos_true, pos_pred, stress_true, stress_pred, node_type_true, node_type_pred, cells

    mask = np.ones(node_type_true.shape[0], dtype=bool)
    if "no_border" in mode: mask &= (node_type_true != BOUNDARY_NODE)
    if "no_sphere" in mode: mask &= (node_type_true != SPHERE_NODE)

    if not mask.any(): raise ValueError("Render mask removed all nodes.")
    
    # Reindex
    idx_map = -np.ones(mask.shape[0], dtype=int)
    keep_idx = np.nonzero(mask)[0]
    idx_map[keep_idx] = np.arange(keep_idx.shape[0])

    # Filter cells
    keep_cells = np.all(mask[cells], axis=1)
    cells_reindexed = idx_map[cells[keep_cells]]

    return (pos_true[mask], pos_pred[mask], 
            stress_true[mask] if stress_true is not None else None, 
            stress_pred[mask] if stress_pred is not None else None,
            node_type_true[mask] if node_type_true is not None else None, 
            node_type_pred[mask] if node_type_pred is not None else None, 
            cells_reindexed)

def _cell_to_edge_index(cells):
    """Helper to reconstruct edge_index from cells for visual logic if needed"""
    # ... (You can paste your _cell_to_edge_index logic here if A is missing) ...
    # But usually A is reconstructed from edge_index in main
    pass

def rollout(model, edge_index, X_seq_norm, mean_vec, std_vec, t0, steps, node_type):
    """
    Rollout function adapted for PyG data structures.
    Arguments:
        edge_index: [2, E] Sparse connectivity
        X_seq_norm: [T, N, 12] Tensor
    """
    device = edge_index.device
    
    # Create Node Type One-Hot for injection
    node_type_onehot = torch.zeros((node_type.shape[0], 2), device=device)
    node_type_onehot[:, 0] = (node_type == SPHERE_NODE).float()
    node_type_onehot[:, 1] = (node_type == BOUNDARY_NODE).float()

    # Masks
    deform_mask = (node_type == NORMAL_NODE)
    rigid_mask = (node_type == SPHERE_NODE)
    border_mask = (node_type == BOUNDARY_NODE)

    # Initial State at t0
    # X_seq_norm is [T, N, 12] -> [Mesh(3), World(3), Type(2), Vel(3), Stress(1)]
    current_norm = X_seq_norm[t0].to(device) # [N, 12]
    
    # Denormalize to get Physics State
    current_phys = current_norm * std_vec + mean_vec
    
    # Trackers
    p_hat = current_phys[:, IDX_WORLD_POS].clone() # Current World Pos [N, 3]
    pos_border_ref = p_hat[border_mask].clone()    # Border reference positions

    coords_pred_list = []
    stress_pred_list = []
    rollout_error_list = []

    for k in range(steps):
        # 1. Prepare Input for Model
        # We need to construct the 15-channel input: [Data(12) + Kinematic(3)]
        
        # Calculate Kinematic Velocity for next step (t0 + k + 1)
        # We cheat and look at GT for the sphere's future velocity
        gt_next_norm = X_seq_norm[t0 + k + 1].to(device)
        gt_next_phys = gt_next_norm * std_vec + mean_vec
        
        v_sphere_next = gt_next_phys[:, IDX_VELOCITY] # Future velocity
        kinematic_vel = torch.zeros_like(v_sphere_next)
        kinematic_vel[rigid_mask] = v_sphere_next[rigid_mask]
        
        # Normalize Kinematic Vel (using velocity stats)
        # Note: Your model might expect normalized kinematic inputs.
        # Assuming we just inject the raw normalized value corresponding to velocity
        kinematic_vel_norm = torch.zeros_like(v_sphere_next)
        kinematic_vel_norm[rigid_mask] = gt_next_norm[rigid_mask, IDX_VELOCITY]

        # Concat: [Current_State(12), Kinematic(3)]
        model_input = torch.cat([current_norm, kinematic_vel_norm], dim=-1) # [N, 15]
        
        # 2. Predict (Normalized)
        with torch.no_grad():
            # Model forward expects (x, edge_index, batch)
            # We pass batch=None since it's a single graph
            pred_norm = model(model_input, edge_index, batch=None) # [N, 4] -> [Vel(3), Stress(1)]

        # 3. Denormalize Predictions
        vel_pred = pred_norm[:, :3] * std_vec[IDX_VELOCITY] + mean_vec[IDX_VELOCITY]
        stress_pred = pred_norm[:, 3:4] * std_vec[IDX_STRESS] + mean_vec[IDX_STRESS]
        
        # 4. Physics Update (Integrate Position)
        p_hat_next = p_hat.clone()
        
        # A. Deformable: p_next = p_prev + v_pred
        p_hat_next[deform_mask] = p_hat[deform_mask] + vel_pred[deform_mask]
        
        # B. Rigid: Force to GT path
        p_hat_next[rigid_mask] = gt_next_phys[rigid_mask, IDX_WORLD_POS]
        vel_pred[rigid_mask]   = gt_next_phys[rigid_mask, IDX_VELOCITY] # Overwrite vel
        stress_pred[rigid_mask] = gt_next_phys[rigid_mask, IDX_STRESS]  # Overwrite stress
        
        # C. Border: Fix position
        p_hat_next[border_mask] = pos_border_ref
        vel_pred[border_mask] = 0.0
        
        # 5. Store & Error
        coords_pred_list.append(p_hat_next.detach().cpu().numpy())
        stress_pred_list.append(stress_pred.detach().cpu().numpy())
        
        mse = torch.mean((p_hat_next - gt_next_phys[:, IDX_WORLD_POS])**2)
        rollout_error_list.append(mse.item())

        # 6. Update State for Next Step
        # Construct new "Current State" vector [N, 12]
        # We reuse MeshPos and NodeType from initial state (they don't change)
        X_next_phys = torch.zeros_like(current_phys)
        
        X_next_phys[:, IDX_MESH_POS] = current_phys[:, IDX_MESH_POS]
        X_next_phys[:, IDX_WORLD_POS] = p_hat_next
        X_next_phys[:, IDX_NODE_TYPE] = current_phys[:, IDX_NODE_TYPE]
        X_next_phys[:, IDX_VELOCITY] = vel_pred
        X_next_phys[:, IDX_STRESS] = stress_pred
        
        # Normalize
        current_norm = (X_next_phys - mean_vec) / std_vec
        current_phys = X_next_phys
        p_hat = p_hat_next

    return coords_pred_list, stress_pred_list, rollout_error_list


def main():
    print("Loading data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Stats
    stats = torch.load(STATS_PATH, map_location=device)
    mean_vec = stats['mean_feat'].to(device)
    std_vec = stats['std_feat'].to(device)
    
    # 2. Load Trajectory File
    traj_path = os.path.join(PREPROCESSED_DIR, f"graph_unet_traj_{TRAJ_INDEX:05d}.pt")
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")
        
    traj_data = torch.load(traj_path, map_location=device)
    
    # Extract Data (Already tensors on device)
    # x_seq is [T, N, 12]
    X_seq_norm = traj_data['x_seq']
    edge_index = traj_data['edge_index'] # [2, E]
    node_type = traj_data['node_type']   # [N]
    
    # We need to recover 'cells' for visualization (triangles)
    # Since we don't save cells in .pt, we must load from original TFRecord OR 
    # if you saved it in .pt (check your preprocess). 
    # Assuming 'cells' is NOT in .pt, we can reconstruct roughly or 
    # BETTER: Just load meta.json or 1 tfrecord to get cells once.
    # Hack: For visualization, if you don't have cells, you can't draw the surface.
    # Did you save cells in pyg_preprocess? 
    # Looking at your preprocess code: NO. You only saved edge_index.
    # **CRITICAL**: Visualization needs cells. 
    # FIX: We will assume you can load 1 sample from TFRecord just to get cells.
    
    # ... (Loading cells from original data for visualization purposes)
    import json
    from tfrecord.reader import tfrecord_loader
    # Quick hack to get cells from the first record of raw data
    print("Fetching cells from raw TFRecord for visualization...")
    loader = tfrecord_loader("data/train.tfrecord", index_path=None)
    first_rec = next(loader)
    cells_bytes = first_rec['cells']
    # You might need your casting logic here if it's raw bytes
    # Assuming standard numpy int32 array for now, adapt if needed based on your decoding logic
    cells = np.frombuffer(cells_bytes, dtype=np.int32).reshape(-1, 4)

    # 3. Load Model
    config = load_config(CONFIG_PATH)
    model_cfg = config['model']
    
    # Input Dim = 15 (12 Data + 3 Kinematic)
    model = GraphUNetDefPlatePyG(
        in_channels=15, 
        hidden_channels=model_cfg['hidden_channels'],
        depth=model_cfg['depth'],
        pool_ratios=model_cfg['pool_ratios'],
        mlp_hidden=model_cfg['mlp_hidden'],
        mlp_dropout=model_cfg.get('mlp_dropout', 0.0)
    ).to(device)
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 4. Run Rollout
    print(f"Starting Rollout from t={T_STEP} for {ROLLOUT_STEPS} steps...")
    
    coords_pred, stress_pred, err_list = rollout(
        model, edge_index, X_seq_norm, mean_vec, std_vec, T_STEP, ROLLOUT_STEPS, node_type
    )
    
    # 5. Visualize
    print("Visualizing...")
    for k in range(ROLLOUT_STEPS):
        t_current = T_STEP + k + 1
        
        # GT at this step
        gt_norm = X_seq_norm[t_current]
        gt_phys = gt_norm * std_vec + mean_vec
        pos_true = gt_phys[:, IDX_WORLD_POS].cpu().numpy()
        stress_true = gt_phys[:, IDX_STRESS].squeeze(-1).cpu().numpy()
        
        # Pred at this step
        pos_p = coords_pred[k]
        stress_p = stress_pred[k].squeeze(-1)
        
        node_type_np = node_type.cpu().numpy()
        
        # Filter (Render Mode)
        p_t, p_p, s_t, s_p, nt_t, nt_p, c_f = apply_render_mode(
            pos_true, pos_p, stress_true, stress_p, node_type_np, node_type_np, cells
        )
        
        visualize_mesh_pair(
            p_t, p_p, c_f, s_t, s_p, nt_t, nt_p,
            f"GT t={t_current}", f"Pred t={t_current}", "stress"
        )
        
    # Plot Error
    fig_err = go.Figure()
    fig_err.add_trace(go.Scatter(y=err_list, mode='lines+markers', name='MSE'))
    fig_err.show()

if __name__ == "__main__":
    main()