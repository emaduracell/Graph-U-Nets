import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GraphUNet


class GraphUNetDefPlatePyG(nn.Module):
    """
    PyG Graph U-Net with separate heads for velocity (3) and stress (1).
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        depth: int,
        pool_ratios,
        mlp_hidden: int,
        mlp_dropout: float = 0.0,
    ):
        super().__init__()

        self.gunet = GraphUNet(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            depth=depth,
            pool_ratios=pool_ratios,
            act=F.relu,
        )

        self.velocity_head = nn.Sequential(
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, 3),
        )

        self.stress_head = nn.Sequential(
            nn.Dropout(mlp_dropout),
            nn.Linear(hidden_channels, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden, 1),
        )

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self.gunet, "reset_parameters"):
            self.gunet.reset_parameters()
        for module in list(self.velocity_head.modules()) + list(self.stress_head.modules()):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, edge_index, batch=None):
        """
        :param x: Node features [num_nodes, in_channels]
        :param edge_index: Edge index [2, num_edges]
        :param batch: Optional batch vector for mini-batching
        :return: predictions [num_nodes, 4] (vx, vy, vz, stress)
        """
        h = self.gunet(x, edge_index, batch=batch)
        vel = self.velocity_head(h)
        stress = self.stress_head(h)
        return torch.cat([vel, stress], dim=-1)


