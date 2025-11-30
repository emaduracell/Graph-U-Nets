import torch
import torch.nn as nn
from src.utils.ops import GCN, GraphUnet, Initializer, norm_g


class GNet_EMA(nn.Module):
    """
    Graph U-Net model for one-step prediction of:
      - node velocities v_{t+1} (3D)
      - node stress σ_{t+1} (scalar)

    Positions x_{t+1} will be reconstructed outside the model as:
        x_pred = x_t + v_pred
    """

    def __init__(self, in_dim, args):
        """
        Parameters
        ----------
        in_dim : int
            Input node feature dimension (F_in). Here: 8.
        args : argparse.Namespace
            Same args used for original GNet:
                act_n   : name of activation for GCNs (e.g. 'ELU')
                act_c   : name of activation for heads (e.g. 'ELU')
                l_dim   : hidden dim in GCN / GraphUnet
                h_dim   : hidden dim in MLP heads
                ks      : list of pool ratios for GraphUnet
                drop_n  : dropout prob for node features (GCN/GraphUnet)
                drop_c  : dropout prob for heads
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

        # Velocity head: [N, l_dim] -> [N, 3]
        self.vel_head = nn.Sequential(
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.l_dim, args.h_dim),
            self.c_act,
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.h_dim, 3),
        )

        # Stress head: [N, l_dim] -> [N, 1]
        self.stress_head = nn.Sequential(
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.l_dim, args.h_dim),
            self.c_act,
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.h_dim, 1),
        )

        # Same initializer as original code
        Initializer.weights_init(self)

    # -------------------------------------------------------- #
    # Public API
    # -------------------------------------------------------- #

    def forward(self, gs, hs):
        """
        Forward over a batch of graphs.

        Parameters
        ----------
        gs : list[Tensor]
            List of adjacency matrices, each of shape [N, N].
        hs : list[Tensor]
            List of input node features at time t, each [N, F_in].

        Returns
        -------
        vel_preds    : Tensor [B, N, 3]   (normalized velocities)
        stress_preds : Tensor [B, N, 1]   (normalized stresses)
        """
        return self.embed(gs, hs)

    def rollout_step(self, g, x_t):
        """
        Single-step prediction, for training/rollouts.

        Parameters
        ----------
        g   : Tensor [N, N]
        x_t : Tensor [N, F_in] (normalized features at time t)

        Returns
        -------
        vel_pred_norm    : Tensor [N, 3]
        stress_pred_norm : Tensor [N, 1]
        """
        vel_pred, stress_pred = self.embed_one(g, x_t)
        return vel_pred, stress_pred

    # -------------------------------------------------------- #
    # Internal helpers
    # -------------------------------------------------------- #

    def embed(self, gs, hs):
        """
        Process a batch of graphs.

        gs : list of [N, N]
        hs : list of [N, F_in]

        Returns:
            vel_preds    : Tensor [B, N, 3]
            stress_preds : Tensor [B, N, 1]
        """
        vel_list = []
        stress_list = []
        for g, h in zip(gs, hs):
            vel_pred, stress_pred = self.embed_one(g, h)  # [N,3], [N,1]
            vel_list.append(vel_pred)
            stress_list.append(stress_pred)

        vel_preds = torch.stack(vel_list, dim=0)        # [B, N, 3]
        stress_preds = torch.stack(stress_list, dim=0)  # [B, N, 1]
        return vel_preds, stress_preds

    def embed_one(self, g, h):
        """
        Process a single graph.

        g : [N, N] adjacency
        h : [N, F_in] node features at time t (normalized)

        Returns:
            vel_pred_norm    : [N, 3]  (normalized velocity)
            stress_pred_norm : [N, 1]  (normalized stress)
        """
        # Normalize adjacency
        g = norm_g(g)  # [N, N]

        # Initial GCN
        h0 = self.s_gcn(g, h)  # [N, l_dim]

        # Graph U-Net: multi-scale node embeddings
        hs = self.g_unet(g, h0)  # list of [N, l_dim]

        # Use final decoder output (full resolution) as node embeddings
        h_nodes = hs[-1]  # [N, l_dim]

        # Two separate heads
        vel_pred = self.vel_head(h_nodes)        # [N, 3]
        stress_pred = self.stress_head(h_nodes)  # [N, 1]

        return vel_pred, stress_pred
