import torch
import torch.nn as nn
from torch_geometric.nn import GraphUNet


class GraphUNet_DefPlate(nn.Module):

    def __init__(self, in_channels, hidden_channels, depth, pool_ratios,
                 model_config_hyperparams):
        """
        PyG GraphUNet wrapper with specific MLP heads for Deforming Plate.

        Args:
            in_channels: Input feature dimension.
            hidden_channels: Hidden dimension for GCN/UNet (l_dim).
            depth: Depth of the U-Net.
            pool_ratios: Pooling ratios for each depth level.
            model_config_hyperparams: object containing:
                - activation_mlps_final (str)
                - dropout_mlps_final (float)
                - hid_mlp_dim (int)
        """
        super(GraphUNet_DefPlate, self).__init__()

        # Extract hyperparameters
        act_name = model_config_hyperparams['activation_mlps_final']
        drop_p = model_config_hyperparams['dropout_mlps_final']
        mlp_hidden = model_config_hyperparams['hid_mlp_dim']

        # Resolve activation function
        self.act_mlps = getattr(nn, act_name)()

        # 1. Backbone: Graph U-Net
        # We set out_channels = hidden_channels to get a latent embedding
        # that we can feed into the MLPs.
        self.unet = GraphUNet(in_channels=in_channels, hidden_channels=hidden_channels, out_channels=hidden_channels,
                              depth=depth, pool_ratios=pool_ratios, sum_res=True, act=nn.ReLU())

        # 2. Velocity MLP Head: [N, hidden] -> [N, 3]
        self.velocity_mlp = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(hidden_channels, mlp_hidden),
            self.act_mlps,
            nn.Dropout(p=drop_p),
            nn.Linear(mlp_hidden, 3),  # Vx, Vy, Vz
        )

        # 3. Stress MLP Head: [N, hidden] -> [N, 1]
        self.stress_mlp = nn.Sequential(
            nn.Dropout(p=drop_p),
            nn.Linear(hidden_channels, mlp_hidden),
            self.act_mlps,
            nn.Dropout(p=drop_p),
            nn.Linear(mlp_hidden, 1),  # Stress
        )

    def forward(self, x_seq, edge_index, batch_idx):
        """
        Args:
            x_seq: Input features. Can be (Batch, T, N, F) or (T, Total_Nodes, F).
            edge_index: (2, E) Connectivity (Mesh + Precomputed World Edges).
            batch_idx: (Total_Nodes) Batch vector.

        Returns:
            out_seq: (T, Total_Nodes, 4) -> [Vel(3), Stress(1)]
        """

        # 1. Handle Input Shapes
        # We want x_seq to be (T, Total_Nodes, F) to loop over time
        if x_seq.dim() == 4:  # Case (Batch, T, N, F)
            B, T, N, F = x_seq.shape
            # Permute to (T, B, N, F) then flatten B*N -> (T, Total_Nodes, F)
            x_seq = x_seq.permute(1, 0, 2, 3).reshape(T, -1, F)

        T = x_seq.shape[0]
        outs = []

        # 2. Temporal Loop
        for t in range(T):
            x_t = x_seq[t]  # (Total_Nodes, F)

            # Backbone: Get Latent Embeddings
            # GraphUNet needs batch_idx for pooling
            latent = self.unet(x_t, edge_index, batch_idx)  # (Total_Nodes, Hidden)

            # Heads
            vel = self.velocity_mlp(latent)  # (Total_Nodes, 3)
            stress = self.stress_mlp(latent)  # (Total_Nodes, 1)

            # Concatenate
            pred_t = torch.cat([vel, stress], dim=-1)
            outs.append(pred_t)

        # Stack back to (T, Total_Nodes, 4)
        return torch.stack(outs, dim=0)