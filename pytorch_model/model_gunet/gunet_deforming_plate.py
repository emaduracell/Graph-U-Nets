import torch
import torch.nn as nn
import torch.nn.functional as F
from model_gunet.gunet_original import GraphUnet
from model_gunet.gunet_nopool import GraphUnetNoPool
from model_gunet.gcn_model import GCN
from model_gunet.initializer import Initializer
from model_gunet.helpers_models import get_adj_norm_fn


class GraphUNet_DefPlate(nn.Module):
    """
    Graph U-Net model for one-step graph-to-graph prediction: (A, X_t) -> (velocity_vec_{t+1}, stress_{t+1})

    self.act_gnn:
        activation function used for the gcn layers
    self.act_mlps_final:
        activation function for the final MLPs for the prediction of stress and velocity
    self.start_gcn:
        initial gcn, which has a dimensionality reduction technique
    self.g_unet:
        graph unet
    self.stress_mlp:
        final MLP for the prediction of stress. It doesn't end with an activation function to avoid enforcing a range
    self.velocity_mlp:
        final MLP for the prediction of velocity. As above for the activation
    self.adj_norm_fn:
        adjacency matrix normalization function
    """

    def __init__(self, in_dim, vel_out_dim, stress_out_dim, model_config_hyperparams, adj_norm):
        """
        Creates an instance of the Graph-U-Net

        Args:
            in_dim: int
                Input node feature dimension (F_in).
            out_dim: int
                Output node feature dimension (F_out).
                Often F_out == F_in if you predict all channels (coords + others).
            model_config_hyperparams: argparse.Namespace
                Same args used for original GNet:
                    act_n   : name of activation for GCNs (e.g. 'ELU')
                    act_c   : name of activation for head (e.g. 'ELU')
                    l_dim   : hidden dim in GCN / GraphUnet
                    h_dim   : hidden dim in node-wise MLP
                    ks      : list of pool ratios for GraphUnet
                    drop_n  : dropout prob for node features (GCN/GraphUnet)
                    drop_c  : dropout prob for head

        :return nothing
        """
        super().__init__()

        # Define local variables from model_config_hyperparams
        act_gnn = model_config_hyperparams.activation_gnn
        act_mlps_final = model_config_hyperparams.activation_mlps_final
        hid_gnn_layer_dim = model_config_hyperparams.hid_gnn_layer_dim
        dropout_gnn = model_config_hyperparams.dropout_gnn
        dropout_mlps_final = model_config_hyperparams.dropout_mlps_final
        hid_mlp_dim = model_config_hyperparams.hid_mlp_dim
        k_pool_ratios = model_config_hyperparams.k_pool_ratios

        self.adj_norm_fn = get_adj_norm_fn(adj_norm)

        # getattr(nn, act_gnn) gets from nn module the activation function with name of the second parameter/string
        self.act_gnn = getattr(nn, act_gnn)()
        self.act_mlps_final = getattr(nn, act_mlps_final)()
        # Initial GCN
        self.start_gcn = GCN(in_dim, hid_gnn_layer_dim, self.act_gnn, dropout_gnn, adj_norm)
        # Graph U-Net
        self.g_unet = GraphUnet(
            ks=k_pool_ratios,
            in_dim=hid_gnn_layer_dim,  # in_dim  (from s_gcn)
            out_dim=hid_gnn_layer_dim,  # out_dim (unused in this impl, kept for API)
            dim=hid_gnn_layer_dim,
            act=self.act_gnn,
            drop_p=dropout_gnn,
            adj_norm=adj_norm
        )
        # self.g_unet = GraphUnetNoPool(
        #     num_layers=len(k_pool_ratios),
        #     in_dim=hid_gnn_layer_dim,  # in_dim  (from s_gcn)
        #     out_dim=hid_gnn_layer_dim,  # out_dim (unused in this impl, kept for API)
        #     dim=hid_gnn_layer_dim,
        #     act=self.act_gnn,
        #     drop_p=dropout_gnn,
        #     adj_norm=adj_norm
        # )
        # Velocity MLP Head: [N, l_dim] -> [N, 3]
        self.velocity_mlp = nn.Sequential(
            nn.Dropout(p=dropout_mlps_final),
            nn.Linear(hid_gnn_layer_dim, hid_mlp_dim),
            self.act_mlps_final,
            nn.Dropout(p=dropout_mlps_final),
            nn.Linear(hid_mlp_dim, vel_out_dim),
        )
        # Stress MLP Head: [N, l_dim] -> [N, 1]
        self.stress_mlp = nn.Sequential(
            nn.Dropout(p=dropout_mlps_final),
            nn.Linear(hid_gnn_layer_dim, hid_mlp_dim),
            self.act_mlps_final,
            nn.Dropout(p=dropout_mlps_final),
            nn.Linear(hid_mlp_dim, stress_out_dim),
        )

        Initializer.weights_init(self)

    def forward(self, batch_adj_A, batch_feat_X):
        """
        Forward over a batch of graphs.

        Args:
            batch_adj_A: list[Tensor]
                List of adjacency matrices, each of shape [N, N].
            batch_feat_X: list[Tensor]
                List of input node features at time t, each [N, F_in].
            feat_tp1_mat_list: Tensor or None
                If provided: tensor of shape [B, N, F_in] with X_{t+1}.
                We only compute loss on velocity (features 4-6) and stress (feature 7).
                Loss is filtered by node_type:
                  - Velocity: only node_type == 0
                  - Stress: node_type == 0 or node_type == 6
                If None: the method returns only predictions.

        :returns
            If targets is not None:
                loss : scalar tensor (MSE)
                preds: Tensor [B, N, 4] (3 velocity + 1 stress)
            If targets is None:
                preds: Tensor [B, N, 4]
        """
        # Prediction
        preds_list = self.embed(batch_adj_A, batch_feat_X)
        return preds_list

    def rollout_step(self, A, X_t):
        """
        Single-step prediction (no loss), for rollouts.

        Args:
            A: Tensor
                [N, N]
            X_t: Tensor
                [N, F_in]

        :return preds_list: Tensor
            [N, F_out]
        """
        return self.embed(A, X_t)

    def embed(self, adj_A_list, X_list):
        """
        Process a batch of graphs, by using embed_one on each.

        Args:
            adj_A_list: List
                list of [N, N] graphs
            X_list: List
                list of [N, F_in]

        :returns preds: Tensor
            Tensor of prediction [B, N, F_out]
        """
        # Single-graph case: tensors
        if isinstance(adj_A_list, torch.Tensor):
            # X_list is then a tensor [N, F_in]
            return self.embed_one(adj_A_list, X_list)
        # Batch case: list of tensors
        return [self.embed_one(A, X) for A, X in zip(adj_A_list, X_list)]

    def embed_one(self, g, h):
        """
        Process a single graph: apply initial GCN, full graph unet, final decoder.

        Args:
            g: [N, N]
                adjacency matrix of the graph
            h: [N, F_in]
                node features at time t

        :returns y_pred: [N, F_out]
            predicted node features at t+1
        """
        # NOTE: adjacency normalization is performed during preprocessing in data_builder.py
        # Initial GCN
        h0 = self.start_gcn(g, h)
        # Graph U-Net
        hs = self.g_unet(g, h0)
        # Use final decoder output as node embeddings
        h_nodes = hs[-1]  # [N, l_dim]
        # Apply separate prediction heads
        vel_pred = self.velocity_mlp(h_nodes)
        stress_pred = self.stress_mlp(h_nodes)
        # Concatenate velocity and stress predictions
        y_pred = torch.cat([vel_pred, stress_pred], dim=-1)

        return y_pred
