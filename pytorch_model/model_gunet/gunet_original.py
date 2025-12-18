import torch
import torch.nn as nn
from model_gunet.helpers_models import get_adj_norm_fn
from model_gunet.gcn_model import GCN
from model_gunet.pool_model import Pool
from model_gunet.unpool_op import Unpool

class GraphUnet(nn.Module):
    """
    Original GraphUnet

    self.ks:
        pooling ratios
    self.bottom_gcn:
        bottom GCN (between end of pooling and start of unpooling)
    self.down_gcns:
        GCNs in the encoding layers
    self.up_gcns:
        GCNs in the decoding layers
    self.pools:
        pooling layers
    self.unpools:
        unpooling layers
    self.l_n:
        number of layers
    self.adj_norm_fn:
        function used to normalize the adjacency matrix
    """

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p, adj_norm):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p, adj_norm)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        self.adj_norm_fn = get_adj_norm_fn(adj_norm)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p, adj_norm=adj_norm))
            self.up_gcns.append(GCN(dim, dim, act, drop_p, adj_norm=adj_norm))
            self.pools.append(Pool(ks[i], dim, drop_p, adj_norm))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        """
        Forward pass in all the GraphUnet.

        Args:
            g: input graph.
            h: ????
        :return: hs prediction
        """
        adj_ms = []
        indices_list = []
        down_outs = []
        hs = []
        org_h = h
        # GCN --> Pool --> repeat (until last pool)
        for i in range(self.l_n):
            h = self.down_gcns[i](g, h)
            adj_ms.append(g)
            down_outs.append(h)
            g, h, idx = self.pools[i](g, h)
            indices_list.append(idx)
        # Bottom GCN before starting going up
        h = self.bottom_gcn(g, h)
        # Unpool --> GCN --> repeat until last GCN
        for i in range(self.l_n):
            up_idx = self.l_n - i - 1
            g, idx = adj_ms[up_idx], indices_list[up_idx]
            g, h = self.unpools[i](g, h, down_outs[up_idx], idx)
            h = self.up_gcns[i](g, h)
            h = h.add(down_outs[up_idx])
            hs.append(h)
        h = h.add(org_h)
        hs.append(h)
        return hs