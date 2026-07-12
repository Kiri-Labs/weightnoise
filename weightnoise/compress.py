"""True SVD compression for transformer models.

Stores truncated U, S, Vt factors per weight matrix instead of full-size W.
The saved model file is actually smaller on disk.

Usage:
    weightnoise compact --model Qwen/Qwen3.5-0.8B --keep-ratio 0.5 --output ./compressed
"""
import torch
import torch.nn as nn


class SVDLinear(nn.Module):
    """Linear layer using truncated SVD factors: output = U @ diag(S) @ Vt @ input.

    Stores factors as nn.Parameter for proper state_dict serialization.
    """

    def __init__(self, in_features, out_features, k, bias=False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.k = k
        self.U = nn.Parameter(torch.empty(out_features, k))
        self.S = nn.Parameter(torch.empty(k))
        self.Vt = nn.Parameter(torch.empty(k, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.U, std=0.01)
        nn.init.ones_(self.S)
        nn.init.normal_(self.Vt, std=0.01)

    def forward(self, x):
        # Wx = U @ diag(S) @ Vt @ x  (without materializing full W)
        vt_x = torch.mm(x, self.Vt.T)
        out = torch.mm(vt_x * self.S.unsqueeze(0), self.U.T)
        if self.bias is not None:
            out = out + self.bias
        return out

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, k={self.k}"


def factorize_weight(W, keep_ratio=0.5, method="svd-observe"):
    """Factorize weight matrix into U, S, Vt with OBS compensation."""
    W = W.float()
    m, n = W.shape
    k = min(m, n)
    k_keep = max(1, int(k * keep_ratio))

    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    if method == "svd-observe":
        saliency = S ** 4
        _, sort_idx = saliency.sort(descending=True)
        keep_mask = torch.zeros(k, dtype=torch.bool)
        keep_mask[sort_idx[:k_keep]] = True

        S_new = S.clone()
        removed = (~keep_mask).nonzero(as_tuple=True)[0]
        kept = keep_mask.nonzero(as_tuple=True)[0]

        if len(removed) > 0:
            U_norm = U / (U.norm(dim=0, keepdim=True) + 1e-8)
            V_norm = Vt.T / (Vt.T.norm(dim=0, keepdim=True) + 1e-8)
            for p in removed:
                coupling = (U_norm[:, kept].T @ U_norm[:, p:p+1]).squeeze() * \
                          (V_norm[:, kept].T @ V_norm[:, p:p+1]).squeeze()
                S_new[kept] += coupling * S[p] / (1 + coupling.abs().mean() + 1e-8)
        return U[:, kept], S_new[kept], Vt[kept, :], len(kept)

    return U[:, :k_keep], S[:k_keep], Vt[:k_keep, :], k_keep


def compress_model(model, keep_ratio=0.5, method="svd-observe"):
    """Convert all Linear layers in a model to SVDLinear.

    Returns (new_model, stats) where stored params are actually smaller.
    """
    stats = {"total_before": 0, "total_after": 0, "converted": 0, "skipped_square": 0}

    def convert_linear(module, name=""):
        for child_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child.weight.ndim == 2:
                m, n = child.weight.shape
                k = min(m, n)
                stats["total_before"] += m * n
                k_kept = max(1, int(k * keep_ratio))
                stored = k_kept * (m + n)

                if stored < m * n:
                    U, S, Vt_k, k_actual = factorize_weight(
                        child.weight.data, keep_ratio, method
                    )
                    svd_lin = SVDLinear(n, m, k_actual, bias=child.bias is not None)
                    svd_lin.U.data = U
                    svd_lin.S.data = S
                    svd_lin.Vt.data = Vt_k
                    if child.bias is not None:
                        svd_lin.bias.data = child.bias.data
                    setattr(module, child_name, svd_lin)
                    stats["converted"] += 1
                    stats["total_after"] += k_actual * (m + n)
                else:
                    stats["total_after"] += m * n
                    stats["skipped_square"] += 1
            else:
                convert_linear(child)

    convert_linear(model)
    cr = stats["total_before"] / max(stats["total_after"], 1)
    stats["cr"] = cr
    return model, stats
