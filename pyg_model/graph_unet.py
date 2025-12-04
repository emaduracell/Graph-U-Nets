from torch_geometric.nn import GraphUNet
import torch
import torch.nn as nn
import torch.nn.functional as F

class DefPlateGraphUnet(nn.Module):
    def __init__(self, in_dim, vel_out_dim, stress_out_dim, model_config_hyperparams):
        super().__init__()

        super().__init__()

        # Define local variables from model_config_hyperparams
        act_gnn = model_config_hyperparams.activation_gnn
        act_mlps_final = model_config_hyperparams.activation_mlps_final
        hid_gnn_layer_dim = model_config_hyperparams.hid_gnn_layer_dim
        dropout_gnn = model_config_hyperparams.dropout_gnn
        dropout_mlps_final = model_config_hyperparams.dropout_mlps_final
        hid_mlp_dim = model_config_hyperparams.hid_mlp_dim
        k_pool_ratios = model_config_hyperparams.k_pool_ratios

        # getattr(nn, act_gnn) gets from nn module the activation function with name of the second parameter/string
        self.act_gnn = getattr(nn, act_gnn)()
        self.act_mlps_final = getattr(nn, act_mlps_final)()

        # Initial GCN
        # self.start_gcn = GCN(in_dim, hid_gnn_layer_dim, self.act_gnn, dropout_gnn)
        # Graph U-Net: TODO NOTE IT DOESN'T HAVE DROPOUT
        self.gunet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            sum_res=True,
            act=act_gnn,
        )
        # Velocity MLP: [N, l_dim] -> [N, 3]
        self.velocity_mlp = nn.Sequential(
            nn.Dropout(p=dropout_mlps_final),
            nn.Linear(hid_gnn_layer_dim, hid_mlp_dim),
            self.act_mlps_final,
            nn.Dropout(p=dropout_mlps_final),
            nn.Linear(hid_mlp_dim, vel_out_dim),
        )
        # Stress MLP: [N, l_dim] -> [N, 1]
        self.stress_mlp = nn.Sequential(
            nn.Dropout(p=dropout_mlps_final),
            nn.Linear(hid_gnn_layer_dim, hid_mlp_dim),
            self.act_mlps_final,
            nn.Dropout(p=dropout_mlps_final),
            nn.Linear(hid_mlp_dim, stress_out_dim),
        )

    def forward(self):
        pass

