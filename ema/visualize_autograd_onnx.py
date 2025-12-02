"""
Visualize the autograd computation graph for Graph_Unet_DefPlate model.

This script creates multiple visualizations of the model architecture:
1. PyTorch autograd graph (forward and backward pass)
2. ONNX graph export for interactive visualization
3. Detailed text architecture description

All visualizations are generated without running actual training.

Requirements:
    pip install -r requirements.txt

Note: You also need to install graphviz system package:
    - macOS: brew install graphviz
    - Ubuntu: sudo apt-get install graphviz
    - Windows: download from https://graphviz.org/download/
"""

import torch
import torch.nn as nn
from torchviz import make_dot
import argparse
import os

from graph_unet_defplate import Graph_Unet_DefPlate


def create_dummy_data(num_nodes, num_features, device):
    """
    Create dummy data that matches the expected input format.
    
    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph
    num_features : int
        Number of input features (default 8: pos[3] + node_type[1] + vel[3] + stress[1])
    device : str
        Device to create tensors on
    
    Returns
    -------
    A : torch.Tensor
        Adjacency matrix [N, N]
    X_t : torch.Tensor
        Input node features [N, F_in]
    X_tp1 : torch.Tensor
        Target node features [N, F_in]
    """
    # Create adjacency matrix (sparse random graph)
    A = torch.zeros((num_nodes, num_nodes), device=device)
    
    # Add some edges (random connectivity)
    num_edges = num_nodes * 3  # roughly 3 edges per node
    edge_indices = torch.randint(0, num_nodes, (2, num_edges))
    A[edge_indices[0], edge_indices[1]] = 1.0
    A[edge_indices[1], edge_indices[0]] = 1.0  # make symmetric
    
    # Add self-loops
    A = A + torch.eye(num_nodes, device=device)
    
    # Row-normalize
    A = A / A.sum(dim=1, keepdim=True)
    
    # Create input features with realistic structure
    # Feature layout: [pos_x, pos_y, pos_z, node_type, vel_x, vel_y, vel_z, stress]
    X_t = torch.randn((num_nodes, num_features), device=device)
    
    # Set node types (feature 3) to 0, 1, or 6
    node_types = torch.randint(0, 3, (num_nodes,), device=device)
    node_types = torch.where(node_types == 2, torch.tensor(6, device=device), node_types)
    X_t[:, 3] = node_types.float()
    
    # Create target (similar structure)
    X_tp1 = X_t + torch.randn_like(X_t) * 0.1  # small perturbation
    
    return A, X_t, X_tp1


def visualize_single_forward_pass(save_path):
    """
    Visualize the forward pass computational graph.
    
    Parameters
    ----------
    save_path : str
        Path to save the visualization
    """
    print("=" * 60)
    print("VISUALIZING FORWARD PASS")
    print("=" * 60)
    
    device = torch.device('cpu')  # Use CPU for visualization
    
    # Create model with dummy arguments
    args = argparse.Namespace()
    args.act_n = 'ELU'
    args.act_c = 'ELU'
    args.l_dim = 32  # smaller dims for clearer visualization
    args.h_dim = 64
    args.ks = [0.9, 0.8]  # fewer pooling levels
    args.drop_n = 0.0  # no dropout for deterministic graph
    args.drop_c = 0.0
    
    F_in = 8
    F_out = 4
    
    model = Graph_Unet_DefPlate(F_in, F_out, args).to(device)
    model.eval()  # Set to eval mode to disable dropout
    
    # Create dummy data
    A, X_t, X_tp1 = create_dummy_data(num_nodes=20, device=device)
    
    # Forward pass through rollout_step
    print("\n1. Running forward pass...")
    pred = model.rollout_step(A, X_t)
    
    print(f"   Input shape: {X_t.shape}")
    print(f"   Output shape: {pred.shape}")
    
    # Create visualization of forward pass
    print("\n2. Creating forward pass visualization...")
    dot = make_dot(pred, params=dict(model.named_parameters()), show_attrs=True, show_saved=False)
    dot.render(save_path.replace('.pdf', ''), format='pdf', cleanup=True)
    print(f"   Saved to: {save_path}")


def visualize_backward_pass(save_path):
    """
    Visualize the full computational graph including backward pass.
    
    Parameters
    ----------
    save_path : str
        Path to save the visualization
    """
    print("\n" + "=" * 60)
    print("VISUALIZING BACKWARD PASS (WITH LOSS)")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Create model
    args = argparse.Namespace()
    args.act_n = 'ELU'
    args.act_c = 'ELU'
    args.l_dim = 32
    args.h_dim = 64
    args.ks = [0.9, 0.8]
    args.drop_n = 0.0
    args.drop_c = 0.0
    
    F_in = 8
    F_out = 4
    
    model = Graph_Unet_DefPlate(F_in, F_out, args).to(device)
    model.train()  # Training mode
    
    # Create dummy data
    A, X_t, X_tp1 = create_dummy_data(num_nodes=20, device=device)
    
    # Forward pass with loss computation (matching training loop)
    print("\n1. Running forward pass with loss computation...")
    pred = model.rollout_step(A, X_t)
    
    # Extract node_type
    node_type = X_t[:, 3]
    
    # Create masks (as in training loop)
    vel_mask = (node_type == 0)
    stress_mask = (node_type == 0) | (node_type == 6)
    
    # Extract targets and predictions
    target_vel = X_tp1[:, 4:7]
    target_stress = X_tp1[:, 7:8]
    pred_vel = pred[:, :3]
    pred_stress = pred[:, 3:4]
    
    # Compute loss
    loss_fn = nn.MSELoss()
    loss = torch.tensor(0.0, device=device, requires_grad=True)
    
    if vel_mask.sum() > 0:
        loss_vel = loss_fn(pred_vel[vel_mask], target_vel[vel_mask])
        loss = loss + loss_vel
    
    if stress_mask.sum() > 0:
        loss_stress = loss_fn(pred_stress[stress_mask], target_stress[stress_mask])
        loss = loss + loss_stress
    
    print(f"   Loss value: {loss.item():.6f}")
    print(f"   Velocity nodes: {vel_mask.sum().item()}/{len(vel_mask)}")
    print(f"   Stress nodes: {stress_mask.sum().item()}/{len(stress_mask)}")
    
    # Create visualization
    print("\n2. Creating backward pass visualization...")
    params_dict = {name: param for name, param in model.named_parameters()}
    params_dict['loss'] = loss
    
    dot = make_dot(loss, params=params_dict, show_attrs=True, show_saved=True)
    dot.render(save_path.replace('.pdf', ''), format='pdf', cleanup=True)
    print(f"   Saved to: {save_path}")


def visualize_detailed_architecture(save_path):
    """
    Create a detailed text representation of the model architecture.
    
    Parameters
    ----------
    save_path : str
        Path to save the text file
    """
    print("\n" + "=" * 60)
    print("MODEL ARCHITECTURE DETAILS")
    print("=" * 60)
    
    # Create model
    args = argparse.Namespace()
    args.act_n = 'ELU'
    args.act_c = 'ELU'
    args.l_dim = 128
    args.h_dim = 256
    args.ks = [0.9, 0.8, 0.7]
    args.drop_n = 0.1
    args.drop_c = 0.1
    
    F_in = 8
    F_out = 4
    
    model = Graph_Unet_DefPlate(F_in, F_out, args)
    
    # Create detailed architecture string
    arch_str = []
    arch_str.append("Graph_Unet_DefPlate Architecture")
    arch_str.append("=" * 60)
    arch_str.append(f"\nInput dimension: {F_in}")
    arch_str.append(f"Output dimension: {F_out}")
    arch_str.append(f"\nHyperparameters:")
    arch_str.append(f"  - l_dim (latent dim): {args.l_dim}")
    arch_str.append(f"  - h_dim (hidden dim): {args.h_dim}")
    arch_str.append(f"  - pooling ratios: {args.ks}")
    arch_str.append(f"  - dropout (nodes): {args.drop_n}")
    arch_str.append(f"  - dropout (classifier): {args.drop_c}")
    arch_str.append(f"\n{'=' * 60}")
    arch_str.append("\nModel Structure:")
    arch_str.append(str(model))
    arch_str.append(f"\n{'=' * 60}")
    arch_str.append("\nParameter Summary:")
    arch_str.append("-" * 60)
    
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
        arch_str.append(f"{name:50s} {str(list(param.shape)):20s} {num_params:,} params")
    
    arch_str.append("-" * 60)
    arch_str.append(f"Total parameters: {total_params:,}")
    arch_str.append(f"Trainable parameters: {trainable_params:,}")
    arch_str.append(f"\n{'=' * 60}")
    arch_str.append("\nData Flow:")
    arch_str.append("-" * 60)
    arch_str.append("Input: X_t [N, 8]")
    arch_str.append("  └─> Feature layout: [pos_x, pos_y, pos_z, node_type, vel_x, vel_y, vel_z, stress]")
    arch_str.append("\n1. Initial GCN (s_gcn):")
    arch_str.append(f"   [N, {F_in}] -> [N, {args.l_dim}]")
    arch_str.append("\n2. Graph U-Net (g_unet):")
    arch_str.append(f"   - Pooling ratios: {args.ks}")
    arch_str.append(f"   - Encoder: [N, {args.l_dim}] -> ... -> [N*k, {args.l_dim}]")
    arch_str.append(f"   - Decoder: [N*k, {args.l_dim}] -> ... -> [N, {args.l_dim}]")
    arch_str.append("\n3. Prediction Heads:")
    arch_str.append("   a) Velocity MLP:")
    arch_str.append(f"      [N, {args.l_dim}] -> [N, {args.h_dim}] -> [N, 3]")
    arch_str.append("      Output: velocity [vx, vy, vz]")
    arch_str.append("      Loss: computed only on nodes with node_type == 0")
    arch_str.append("\n   b) Stress MLP:")
    arch_str.append(f"      [N, {args.l_dim}] -> [N, {args.h_dim}] -> [N, 1]")
    arch_str.append("      Output: stress [σ]")
    arch_str.append("      Loss: computed only on nodes with node_type ∈ {0, 6}")
    arch_str.append("\n4. Output:")
    arch_str.append("   Concatenated: [N, 4] = [velocity (3) + stress (1)]")
    arch_str.append(f"\n{'=' * 60}")
    
    # Write to file
    with open(save_path, 'w') as f:
        f.write('\n'.join(arch_str))
    
    print('\n'.join(arch_str))
    print(f"\nArchitecture details saved to: {save_path}")


def export_onnx_graph(save_path):
    """
    Export the model to ONNX format for visualization.
    
    Parameters
    ----------
    save_path : str
        Path to save the ONNX file
    """
    print("\n" + "=" * 60)
    print("EXPORTING ONNX GRAPH")
    print("=" * 60)
    
    device = torch.device('cpu')
    
    # Create model
    args = argparse.Namespace()
    args.act_n = 'ELU'
    args.act_c = 'ELU'
    args.l_dim = 32  # smaller dims for clearer visualization
    args.h_dim = 64
    args.ks = [0.9, 0.8]
    args.drop_n = 0.0
    args.drop_c = 0.0
    
    F_in = 8
    F_out = 4
    
    model = Graph_Unet_DefPlate(F_in, F_out, args).to(device)
    model.eval()
    
    # Create dummy input
    A, X_t, _ = create_dummy_data(num_nodes=20, device=device)
    
    print("\n1. Preparing model for ONNX export...")
    print(f"   Input shapes: A={A.shape}, X_t={X_t.shape}")
    
    # ONNX export requires a forward pass wrapper since our model uses list inputs
    class ONNXWrapper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
        
        def forward(self, A, X_t):
            # Call embed_one directly for single graph
            return self.model.embed_one(A, X_t)
    
    wrapped_model = ONNXWrapper(model)
    
    print("\n2. Exporting to ONNX format...")
    try:
        torch.onnx.export(
            wrapped_model,
            (A, X_t),
            save_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=['adjacency_matrix', 'node_features'],
            output_names=['predictions'],
            dynamic_axes={
                'adjacency_matrix': {0: 'num_nodes', 1: 'num_nodes'},
                'node_features': {0: 'num_nodes'},
                'predictions': {0: 'num_nodes'}
            },
            verbose=False
        )
        print(f"   ✓ Successfully exported to: {save_path}")
        print(f"   File size: {os.path.getsize(save_path) / 1024:.1f} KB")
        
        # Verify the ONNX model
        print("\n3. Verifying ONNX model...")
        try:
            import onnx
            onnx_model = onnx.load(save_path)
            onnx.checker.check_model(onnx_model)
            print("   ✓ ONNX model is valid")
            
            # Print model info
            print(f"\n   Model inputs:")
            for input in onnx_model.graph.input:
                print(f"     - {input.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in input.type.tensor_type.shape.dim]}")
            print(f"\n   Model outputs:")
            for output in onnx_model.graph.output:
                print(f"     - {output.name}: {[d.dim_value if d.dim_value > 0 else 'dynamic' for d in output.type.tensor_type.shape.dim]}")
            
            print(f"\n   Graph contains {len(onnx_model.graph.node)} nodes (operations)")
            
        except ImportError:
            print("   ⚠ onnx package not found. Install with: pip install onnx")
            print("   (ONNX export still successful, just can't verify)")
        except Exception as e:
            print(f"   ⚠ ONNX verification warning: {e}")
    
    except Exception as e:
        print(f"   ✗ ONNX export failed: {e}")
        return False
    
    print("\n4. Visualization options:")
    print("   You can visualize the ONNX graph using:")
    print("   a) Netron (recommended): https://netron.app")
    print(f"      Open {save_path} in Netron to see interactive graph")
    print("   b) Command line: pip install netron && netron " + save_path)
    print("   c) Python: import netron; netron.start('" + save_path + "')")
    
    return True


def main():
    """Main visualization function."""
    print("\n" + "=" * 60)
    print("AUTOGRAD GRAPH VISUALIZATION FOR Graph_Unet_DefPlate")
    print("=" * 60)
    
    try:
        import torchviz
        print("\n✓ torchviz found")
    except ImportError:
        print("\n✗ torchviz not found. Install with: pip install torchviz")
        print("  Also install graphviz system package:")
        print("    - macOS: brew install graphviz")
        print("    - Ubuntu: sudo apt-get install graphviz")
        return
    
    # 1. Visualize forward pass
    visualize_single_forward_pass('model_graphs/autograd_forward.pdf')
    
    # 2. Visualize backward pass with loss
    visualize_backward_pass('model_graphs/autograd_backward.pdf')
    
    # 3. Create detailed architecture text
    visualize_detailed_architecture('model_architecture.txt')
    
    # 4. Export ONNX graph
    export_onnx_graph('model_graphs/model_graph.onnx')
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE!")
    print("=" * 60)
    print("\nGenerated files:")
    print("  1. autograd_forward.pdf     - Forward pass computational graph")
    print("  2. autograd_backward.pdf    - Backward pass with loss and gradients")
    print("  3. model_architecture.txt   - Detailed model architecture")
    print("  4. model_graph.onnx         - ONNX format (view in Netron)")
    print("\nYou can view these files to see the complete autograd graph")
    print("without running the actual training.")
    print("\nFor interactive ONNX visualization:")
    print("  - Visit https://netron.app and open model_graph.onnx")
    print("  - Or run: pip install netron && netron model_graph.onnx")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

