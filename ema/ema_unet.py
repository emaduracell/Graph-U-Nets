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

        # Node-wise prediction head:
        # takes node embeddings [N, l_dim] -> [N, out_dim]
        self.node_mlp = nn.Sequential(
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.l_dim, args.h_dim),
            self.c_act,
            nn.Dropout(p=args.drop_c),
            nn.Linear(args.h_dim, out_dim),
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
            If provided: tensor of shape [B, N, F_out] with X_{t+1}.
            If None: the method returns only predictions.

        Returns
        -------
        If targets is not None:
            loss : scalar tensor (MSE)
            preds: Tensor [B, N, F_out]
        If targets is None:
            preds: Tensor [B, N, F_out]
        """
        preds = self.embed(gs, hs)  # [B, N, F_out]

        if targets is None:
            return preds

        loss = F.mse_loss(preds, targets)
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

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

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

        # Node-wise regression head
        y_pred = self.node_mlp(h_nodes)  # [N, F_out]

        return y_pred
