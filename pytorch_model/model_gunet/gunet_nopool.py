import torch
import torch.nn as nn
from model_gunet.helpers_models import get_adj_norm_fn
from model_gunet.gcn_model import GCN


class GraphUnetNoPool(nn.Module):
    """
    Graph U-Net architecture without pooling/unpooling.
    Just encoder GCNs, bottom GCN, decoder GCNs with skip connections.
    """

    def __init__(self, num_layers, in_dim, out_dim, dim, act, drop_p, adj_norm):
        super().__init__()
        self.bottom_gcn = GCN(dim, dim, act, drop_p, adj_norm)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.l_n = num_layers

        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p, adj_norm=adj_norm))
            self.up_gcns.append(GCN(dim, dim, act, drop_p, adj_norm=adj_norm))

    def forward(self, g, h):
        down_outs = []
        hs = []
        org_h = h

        # Encoder
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            down_outs.append(h)

        # Bottom
        h = self.bottom_gcn(g, h)

        # Decoder with skip connections
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            h = h + down_outs[up_idx]  # Skip connection (simple addition)
            h = self.up_gcns[i](g, h)
            hs.append(h)

        # Final residual from input
        h = h + org_h
        hs.append(h)
        return hs