import torch
import torch.nn as nn
import numpy as np


class GraphUnet(nn.Module):
    """
    Original GraphUnet

    self.ks: pooling ratios
    self.bottom_gcn: bottom GCN (between end of pooling and start of unpoolin)
    self.down_gcns: GCNs in the encoding layers
    self.up_gcns: GCNs in the decoding layers
    self.pools: pooling layers
    self.unpools: unpooling layers
    self.l_n: number of layers
    """

    def __init__(self, ks, in_dim, out_dim, dim, act, drop_p):
        super(GraphUnet, self).__init__()
        self.ks = ks
        self.bottom_gcn = GCN(dim, dim, act, drop_p)
        self.down_gcns = nn.ModuleList()
        self.up_gcns = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.unpools = nn.ModuleList()
        self.l_n = len(ks)
        for i in range(self.l_n):
            self.down_gcns.append(GCN(dim, dim, act, drop_p))
            self.up_gcns.append(GCN(dim, dim, act, drop_p))
            self.pools.append(Pool(ks[i], dim, drop_p))
            self.unpools.append(Unpool(dim, dim, drop_p))

    def forward(self, g, h):
        """
        Forward pass in all the GraphUnet.

        :param g: input graph.
        :param h: ????
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


class GCN(nn.Module):
    """
    Graph Convolutional Network: takes care of message passing (aggregating node neighborhoods).

    self.proj: linear layer (trainable)
    self.act: activation function
    self.drop: dropout
    """

    def __init__(self, in_dim, out_dim, act, p):
        super(GCN, self).__init__()
        self.proj = nn.Linear(in_dim, out_dim)  # learnable
        self.act = act
        self.drop = nn.Dropout(p=p) if p > 0.0 else nn.Identity()

    def forward(self, g, h):
        """
        Forward pass in GCN. Dropout --> Convolution/Aggregation (matmul) --> learnable linear layer --> activation

        :param g:
            adjacency matrix
        :param h:
            embedded matrix until here

        :return: h
            resulting new prediction/embedded matrix
        """
        h = self.drop(h)
        h = torch.matmul(g, h)  # convolution step
        h = self.proj(h)  # learnable
        h = self.act(h)
        return h


class Pool(nn.Module):
    """
    Layer that takes care of pooling, that is, a scalar projection.

    self.k:
        pooling ratio (how many nodes to keep)
    self.sigmoid:
        activation function
    self.proj:
        linear projection (trainable)
    self.drop:
        dropout
    """

    def __init__(self, k, in_dim, p):
        super(Pool, self).__init__()
        self.k = k
        self.sigmoid = nn.Sigmoid()
        self.proj = nn.Linear(in_dim, 1)
        self.drop = nn.Dropout(p=p) if p > 0 else nn.Identity()

    def forward(self, g, h):
        """
        Forward pass of Pooling layer: Dropout layer --> Linear layer learnable (out: scalar score for each node, or
        better for each node's feature vector) --> sigmoid (activation)

        :param g:
            adjacency matrix
        :param h:
            embedded matrix until here

        :return: top_k_graph(scores, g, h, self.k)
            resulting new prediction/embedded matrix
        """
        Z = self.drop(h)
        weights = self.proj(Z).squeeze()  # learnable
        scores = self.sigmoid(weights)
        return top_k_graph(scores, g, h, self.k)


class Unpool(nn.Module):
    """
    Unpooling class, not learnable.
    """

    def __init__(self, *args):
        super(Unpool, self).__init__()

    def forward(self, g, h, pre_h, idx):
        """
        1. Returns a new matrix

        :param g:
            original graph adjacency matrix (skip connection)
        :param h:
            input (encoded) embedded feature matrix
        :param pre_h:
            NOT IMPLEMENTED
        :param idx:
            indexes of previously kept nodes by top-k.

        :return: (g, new_h)
            g: original graph adjacency matrix (skip connection)
            new_h: new feature matrix
        """
        new_h = h.new_zeros([g.shape[0], h.shape[1]])
        new_h[idx] = h
        return g, new_h


def top_k_graph(scores, g, h, k):
    """
    Picks the top k nodes in the graph, recomputes adjacency matrix.
    1. Computes indexes and score of top k nodes (idx, values)
    2. Reweights every selected node feature vector by the score of the node (values)
    3. I take the (normalized) adjacency matrix, convert it to 0,1 entries, square it (2-path connectivity hop).
    4. I re-convert the result of 1 to 0,1 entries
    5. Take the adjacency matrix of the subgraph of idx.
    6. Row-normalizes the result of 5 with norm_g.

    :param scores:
        number of nodes
    :param g:
        adjacency matrix of the graph
    :param h:
        input embedded matrix until this point
    :param k:
        number of nodes to keep

    :return: (g, new_h, idx)
        g: new adjacency matrix
        new_h: new embedded layer matrix
        idx: index of selected nodes
    """
    num_nodes = g.shape[0]
    # Following line:
    values_score, idx = torch.topk(scores, max(2, int(k * num_nodes)))
    new_h = h[idx, :]
    values_score = torch.unsqueeze(values_score, -1)
    new_h = torch.mul(new_h, values_score)
    un_g = g.bool().float()
    un_g = torch.matmul(un_g, un_g).bool().float()
    un_g = un_g[idx, :]
    un_g = un_g[:, idx]
    g = norm_g(un_g)
    return g, new_h, idx


def norm_g(g):
    """
    Row-normalizes the adjacency matrix.
    1. Sums across columns, that is, gives a vector where element i is the sum of all elements of row i.
    2. Divides every row i of the adjacency matrix by degrees[i]

    :param g:
        adjacency matrix of the graph

    :return: g
        new row-normalized adjacency matrix
    """
    degrees = torch.sum(g, dim=1)
    g = g / degrees
    return g


class Initializer(object):
    """
    Class that initializes.
    """

    @classmethod
    def _glorot_uniform(cls, w):
        if len(w.size()) == 2:
            fan_in, fan_out = w.size()
        elif len(w.size()) == 3:
            fan_in = w.size()[1] * w.size()[2]
            fan_out = w.size()[0] * w.size()[2]
        else:
            fan_in = np.prod(w.size())
            fan_out = np.prod(w.size())
        limit = np.sqrt(6.0 / (fan_in + fan_out))
        w.uniform_(-limit, limit)

    @classmethod
    def _param_init(cls, m):
        if isinstance(m, nn.parameter.Parameter):
            cls._glorot_uniform(m.data)
        elif isinstance(m, nn.Linear):
            m.bias.data.zero_()
            cls._glorot_uniform(m.weight.data)

    @classmethod
    def weights_init(cls, m):
        for p in m.modules():
            if isinstance(p, nn.ParameterList):
                for pp in p:
                    cls._param_init(pp)
            else:
                cls._param_init(p)

        for name, p in m.named_parameters():
            if '.' not in name:
                cls._param_init(p)
