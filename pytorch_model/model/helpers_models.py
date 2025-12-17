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
    
    if un_g.is_sparse:
        print(f"[top_k_graph] un_g is sparse, shape: {un_g.shape}")
        un_g = un_g.to_dense()
        
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

    :param g:
        adjacency matrix of the graph

    :return: g
        new row-normalized adjacency matrix
    """
    if g.is_sparse:
        print(f"[norm_g] g is sparse, shape: {g.shape}")
        # Sparse row normalization
        g = g.coalesce()
        degrees = torch.sparse.sum(g, dim=1).to_dense() # [N]
        degrees = degrees.clamp(min=1e-12)
        
        # In sparse COO, we divide values by degree of row index
        indices = g.indices() # [2, E]
        row_indices = indices[0]
        values = g.values()
        
        new_values = values / degrees[row_indices]
        return torch.sparse_coo_tensor(indices, new_values, g.shape).coalesce()
        
    degrees = torch.sum(g, dim=1, keepdim=True)
    degrees = degrees.clamp(min=1e-12)
    g = g / degrees
    return g

def norm_adj_sym(A: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}
    A: [N, N] or [B, N, N]
    """
    if A.is_sparse:
        print(f"[norm_adj_sym] A is sparse, shape: {A.shape}")
        A = A.coalesce()
        if A.dim() != 2:
            raise ValueError(f"Sparse A must be 2D, got shape {tuple(A.shape)}")
            
        deg = torch.sparse.sum(A, dim=1).to_dense()
        inv_sqrt_deg = (deg.clamp_min(eps)).pow(-0.5)
        
        indices = A.indices()
        row_indices = indices[0]
        col_indices = indices[1]
        values = A.values()
        
        # D^-1/2 * A * D^-1/2 -> A_ij / (sqrt(deg_i) * sqrt(deg_j))
        new_values = values * inv_sqrt_deg[row_indices] * inv_sqrt_deg[col_indices]
        return torch.sparse_coo_tensor(indices, new_values, A.shape).coalesce()

    deg = A.sum(dim=-1)                              # [N] or [B, N]
    inv_sqrt_deg = (deg.clamp_min(eps)).pow(-0.5)    # avoid inf for deg=0

    if A.dim() == 2:
        return inv_sqrt_deg[:, None] * A * inv_sqrt_deg[None, :]
    elif A.dim() == 3:
        return inv_sqrt_deg[:, :, None] * A * inv_sqrt_deg[:, None, :]
    else:
        raise ValueError(f"A must be 2D or 3D, got shape {tuple(A.shape)}")

def get_adj_norm_fn(adj_norm: str):
    if adj_norm == "sym":
        return norm_adj_sym
    if adj_norm == "row":
        return norm_g
    raise ValueError(f"Normalization of adjacency matrix = {adj_norm} not implemented")