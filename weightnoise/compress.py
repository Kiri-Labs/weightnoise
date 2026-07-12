"""True SVD compression for transformer models.

Stores truncated U, S, Vt factors per weight matrix instead of full-size W.
The saved model file is actually smaller on disk.

Usage:
    weightnoise compress --model Qwen/Qwen3.5-0.8B --keep-ratio 0.5 --save-path ./compressed
    weightnoise decompress --model ./compressed --save-path ./restored
"""
import torch
import torch.nn as nn
import os, json
from transformers import AutoModelForCausalLM, AutoConfig
from huggingface_hub import snapshot_download


class SVDLinear(nn.Module):
    """Linear layer using truncated SVD factors: output = U @ diag(S) @ Vt @ input.

    Stores factors as nn.Parameter for proper state_dict serialization.
    Parameters: U(m x k), S(k,), Vt(k x n) instead of W(m x n).
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
        # Compute W @ x via factors: U @ (S.unsqueeze(-1) * (Vt @ x.T)).T
        # Vt @ x.T -> (k, in) @ (batch, in).T -> (k, batch)
        # S[:, None] * ... -> (k, batch)
        # U @ ... -> (out, k) @ (k, batch) -> (out, batch)
        # Then transpose to (batch, out)
        vt_x = torch.mm(x, self.Vt.T)  # (batch, k)
        u_s_vt_x = torch.mm(vt_x * self.S.unsqueeze(0), self.U.T)  # (batch, out)
        if self.bias is not None:
            u_s_vt_x = u_s_vt_x + self.bias
        return u_s_vt_x

    def extra_repr(self):
        return f"in={self.in_features}, out={self.out_features}, k={self.k}"


def factorize_weight(W, keep_ratio=0.5, method="svd-observe"):
    """Factorize weight matrix into U, S, Vt with OBS compensation.

    Returns (U, S, Vt, k_keep) where storing these saves k*(m+n) vs m*n floats.
    """
    W = W.float()
    m, n = W.shape
    k = min(m, n)
    k_keep = max(1, int(k * keep_ratio))

    U, S, Vt = torch.linalg.svd(W, full_matrices=False)

    if method == "svd-observe":
        # OBS compensation: redistribute removed SVs to kept ones
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

    # Plain SVD: just keep top-k
    return U[:, :k_keep], S[:k_keep], Vt[:k_keep, :], k_keep


def compress_model(model, keep_ratio=0.5, method="svd-observe", layer_filter=None):
    """Convert all Linear layers in a model to SVDLinear.

    Returns (new_model, stats) where new_model has actually smaller parameters.
    """
    stats = {"total_before": 0, "total_after": 0, "converted": 0, "skipped_square": 0}

    def convert_linear(module, name=""):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name

            if isinstance(child, nn.Linear) and child.weight.ndim == 2:
                m, n = child.weight.shape
                k = min(m, n)
                stats["total_before"] += m * n
                k_kept = max(1, int(k * keep_ratio))
                stored = k_kept * (m + n)
                stats["total_after"] += stored

                # Only convert if actual savings (rectangular or high compression)
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
                else:
                    stats["skipped_square"] += 1
            else:
                convert_linear(child, full_name)

    convert_linear(model)
    cr = stats["total_before"] / max(stats["total_after"], 1)
    stats["cr"] = cr
    return model, stats


def decompress_model(model):
    """Restore SVDLinear layers back to standard Linear (full weight reconstruction)."""
    def restore_linear(module):
        for child_name, child in list(module.named_children()):
            if isinstance(child, SVDLinear):
                # Reconstruct W = U @ diag(S) @ Vt
                W = child.U @ (child.S.unsqueeze(-1) * child.Vt)
                lin = nn.Linear(child.in_features, child.out_features, bias=child.bias is not None)
                lin.weight.data = W
                if child.bias is not None:
                    lin.bias.data = child.bias.data
                setattr(module, child_name, lin)
            else:
                restore_linear(child)
    decompress_model(model)
    return model


def load_compressed_model(path, device="cpu"):
    """Load a compressed model with SVDLinear layers for inference."""
    from transformers import AutoConfig, AutoTokenizer
    # Load the config first, then instantiate a standard model
    config = AutoConfig.from_pretrained(path)
    # We need to load the state dict which has different shapes than standard
    # This requires knowing the original architecture and patching
    # For now: load and decompress on-the-fly
    state_dict = torch.load(os.path.join(path, "pytorch_model.bin"), map_location=device)
    # Reconstruct standard model then load
    model = AutoModelForCausalLM.from_config(config)
    # Map SVDLinear params to standard Linear
    new_sd = {}
    for key, tensor in state_dict.items():
        if key.endswith(".U"):
            base = key[:-2]
            if f"{base}.S" in state_dict and f"{base}.Vt" in state_dict:
                U = tensor
                S = state_dict[f"{base}.S"]
                Vt = state_dict[f"{base}.Vt"]
                new_sd[f"{base}.weight"] = U @ (S.unsqueeze(-1) * Vt)
            else:
                new_sd[key] = tensor
        elif key.endswith(".S") or key.endswith(".Vt"):
            continue
        else:
            new_sd[key] = tensor
    model.load_state_dict(new_sd, strict=False)
    model.eval()
    return model
