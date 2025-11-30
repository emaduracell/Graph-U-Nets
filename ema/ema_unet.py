import torch
import torch.nn as nn
import torch.nn.functional as F
from src.utils.ops import GCN, GraphUnet, Initializer, norm_g


class GNet_EMA(nn.Module):
    """
    Graph U-Net model for one-step graph-to-graph prediction:
        (A, X_t) -> X_{t+1}

    Differences vs original GNet:
      - no graph-level readout
      - no classification head
      - node-wise regression head to predict next-step features
    """

    def __init__(self, in_dim, out_dim, args):
        """
        Parameters
        ----------
        in_dim : int
            Input node feature dimension (F_in).
        out_dim : int
            Output node feature dimension (F_out).
            Often F_out == F_in if you predict all channels (coords + others).
        args : argparse.Namespace
            Same args used for original GNet:
                act_n   : name of activation for GCNs (e.g. 'ELU')
                act_c   : name of activation for head (e.g. 'ELU')
                l_dim   : hidden dim in GCN / GraphUnet
                h_dim   : hidden dim in node-wise MLP
                ks      : list of pool ratios for GraphUnet
                drop_n  : dropout prob for node features (GCN/GraphUnet)
                drop_c  : dropout prob for head
        """
        super().__init__()

        # Activations (same style as original GNet)
        self.n_act = getattr(nn, args.act_n)()
        self.c_act = getattr(nn, args.act_c)()

        # Initial GCN mapping input features -> GraphUnet feature dim
        self.s_gcn = GCN(in_dim, args.l_dim, self.n_act, args.drop_n)

        # Graph U-Net backbone (unchanged)
        self.g_unet = GraphUnet(
            args.ks,          # pooling ratios
            args.l_dim,       # in_dim  (from s_gcn)
            args.l_dim,       # out_dim (unused in this impl, kept for API)
            args.l_dim,       # internal feature dim
            self.n_act,
            args.drop_n
        )

        # Separate prediction heads for velocity (3D) and stress (1D)
        # Velocity MLP: [N, l_dim] -> [N, 3]
        self.velocity_mlp = nn.Sequential(
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.l_dim, args.h_dim),
            self.c_act,
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.h_dim, 3),
        )
        
        # Stress MLP: [N, l_dim] -> [N, 1]
        self.stress_mlp = nn.Sequential(
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.l_dim, args.h_dim),
            self.c_act,
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.h_dim, 1),
        )

        # Same initializer as original code
        Initializer.weights_init(self)


    def forward(self, gs, hs, targets=None):
        """
        Forward over a batch of graphs.

        Parameters
        ----------
        gs : list[Tensor]
            List of adjacency matrices, each of shape [N, N].
        hs : list[Tensor]
            List of input node features at time t, each [N, F_in].
        targets : Tensor or None
            If provided: tensor of shape [B, N, F_in] with X_{t+1}.
            We only compute loss on velocity (features 4-6) and stress (feature 7).
            Loss is filtered by node_type:
              - Velocity: only node_type == 0
              - Stress: node_type == 0 or node_type == 6
            If None: the method returns only predictions.

        Returns
        -------
        If targets is not None:
            loss : scalar tensor (MSE)
            preds: Tensor [B, N, 4] (3 velocity + 1 stress)
        If targets is None:
            preds: Tensor [B, N, 4]
        """
        preds = self.embed(gs, hs)  # [B, N, 4] (velocity + stress)

        if targets is None:
            return preds

        # Extract node_type from input features (feature index 3)
        hs_tensor = torch.stack(hs, dim=0)  # [B, N, F_in]
        node_types = hs_tensor[:, :, 3]  # [B, N]
        
        # Create masks for filtering
        # Velocity: only node_type == 0
        vel_mask = (node_types == 0)  # [B, N]
        # Stress: node_type == 0 or node_type == 6
        stress_mask = (node_types == 0) | (node_types == 6)  # [B, N]
        
        # Extract targets
        target_vel = targets[:, :, 4:7]    # [B, N, 3]
        target_stress = targets[:, :, 7:8] # [B, N, 1]
        
        # Extract predictions
        pred_vel = preds[:, :, :3]          # [B, N, 3]
        pred_stress = preds[:, :, 3:4]      # [B, N, 1]
        
        # Compute separate losses with masks
        loss = 0.0
        
        # Velocity loss (filtered by vel_mask)
        if vel_mask.sum() > 0:
            vel_mask_expanded = vel_mask.unsqueeze(-1).expand_as(pred_vel)  # [B, N, 3]
            loss_vel = F.mse_loss(pred_vel[vel_mask_expanded], target_vel[vel_mask_expanded])
            loss = loss + loss_vel
        
        # Stress loss (filtered by stress_mask)
        if stress_mask.sum() > 0:
            stress_mask_expanded = stress_mask.unsqueeze(-1).expand_as(pred_stress)  # [B, N, 1]
            loss_stress = F.mse_loss(pred_stress[stress_mask_expanded], target_stress[stress_mask_expanded])
            loss = loss + loss_stress
        
        return loss, preds

    
    def rollout_step(self, g, x_t):
        """
        Single-step prediction (no loss), for rollouts.

        Parameters
        ----------
        g   : Tensor [N, N]
        x_t : Tensor [N, F_in]

        Returns
        -------
        x_pred : Tensor [N, F_out]
        """
        return self.embed([g], [x_t])[0]
    # Internal helpers

    def embed(self, gs, hs):
        """
        Process a batch of graphs.

        gs : list of [N, N]
        hs : list of [N, F_in]

        Returns:
            preds : Tensor [B, N, F_out]
        """
        out_list = []
        for g, h in zip(gs, hs):
            out = self.embed_one(g, h)  # [N, F_out]
            out_list.append(out)
        preds = torch.stack(out_list, dim=0)  # [B, N, F_out]
        return preds

    def embed_one(self, g, h):
        """
        Process a single graph.

        g : [N, N] adjacency
        h : [N, F_in] node features at time t

        Returns:
            y_pred : [N, F_out] predicted node features at t+1
        """
        # Normalize adjacency
        g = norm_g(g)  # [N, N]

        # Initial GCN
        h0 = self.s_gcn(g, h)  # [N, l_dim]

        # Graph U-Net: multi-scale node embeddings
        hs = self.g_unet(g, h0)  # list of [N, l_dim]

        # Use final decoder output (full resolution) as node embeddings
        h_nodes = hs[-1]  # [N, l_dim]

        # Apply separate prediction heads
        vel_pred = self.velocity_mlp(h_nodes)  # [N, 3]
        stress_pred = self.stress_mlp(h_nodes)  # [N, 1]
        
        # Concatenate velocity and stress predictions
        y_pred = torch.cat([vel_pred, stress_pred], dim=-1)  # [N, 4]

        return y_pred
