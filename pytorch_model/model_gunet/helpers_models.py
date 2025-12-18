import torch

def top_k_graph(scores, g, h, k, adj_norm_fn):
    """
    Picks the top k nodes in the graph, recomputes adjacency matrix.
    1. Computes indexes and score of top k nodes (idx, values)
    2. Reweights every selected node feature vector by the score of the node (values)
    3. I take the (normalized) adjacency matrix, convert it to 0,1 entries, square it (2-path connectivity hop).
    4. I re-convert the result of 1 to 0,1 entries
    5. Take the adjacency matrix of the subgraph of idx.
    6. Row-normalizes the result of 5 with norm_g.

    Args:
        scores:
            number of nodes
        g:
            adjacency matrix of the graph
        h:
            input embedded matrix until this point
        k:
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
    g = adj_norm_fn(un_g)
    return g, new_h, idx


def norm_g(g):
    """
    Row-normalizes the adjacency matrix.
    1. Sums across columns, that is, gives a vector where element i is the sum of all elements of row i.
    2. Divides every row i of the adjacency matrix by degrees[i]

    Args:
        g:
            adjacency matrix of the graph

    :return: g
        new row-normalized adjacency matrix
    """

    degrees = torch.sum(g, dim=-1, keepdim=True)
    degrees = degrees.clamp(min=1e-12)
    g = g / degrees
    return g

def norm_adj_sym(A, eps):
    """
    Symmetric normalization

    Args:
        A: [N, N] or [B, N, N]
        eps: avoids division by zero
    :return normalized matrix
    """
    deg = A.sum(dim=-1)
    inv_sqrt_deg = (deg.clamp_min(eps)).pow(-0.5)

    if A.dim() == 2:
        return inv_sqrt_deg[:, None] * A * inv_sqrt_deg[None, :]
    elif A.dim() == 3:
        return inv_sqrt_deg[:, :, None] * A * inv_sqrt_deg[:, None, :]
    else:
        raise ValueError(f"A must be 2D or 3D, got shape {tuple(A.shape)}")

def get_adj_norm_fn(adj_norm):
    """
    Returns the proper callable function, given the input parameter
    Args:
        adj_norm: str
            desired normalization functrion
    :return: callable function
    """
    if adj_norm == "sym":
        return norm_adj_sym
    if adj_norm == "row":
        return norm_g
    raise ValueError(f"Normalization of adjacency matrix = {adj_norm} not implemented")