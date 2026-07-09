"""Pruning engine: remove noise from model weights."""
import torch
import numpy as np
import os
import re
from transformers import AutoModelForCausalLM


class NoisePruner:
    """Remove noise from model weights using various methods."""

    def __init__(self, model_id: str, device: str = "cpu", trust_remote_code: bool = False):
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code

    def prune(self, method: str = "magnitude", keep_ratio: float = 0.5,
              threshold: float = 0.01, save_path: str = None):
        """Prune noise from model weights.

        Args:
            method: 'magnitude' — zero out smallest |w| weights
                    'spectral' — zero out weights corresponding to small singular values
                    'wanda' — weight × activation norm importance (needs calibration data)
            keep_ratio: Fraction of weights to keep (0.5 = remove 50%)
            threshold: Spectral noise threshold (fraction of max singular value)
            save_path: Path to save pruned model

        Returns:
            dict with pruning stats
        """
        print(f"  Loading {self.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )

        original_params = sum(p.numel() for p in model.parameters())
        total_removed = 0
        stats = {}

        print(f"  Pruning with {method} method...")
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.ndim != 2:
                    continue

                w = param.float()
                m, n = w.shape
                k = min(m, n)

                if method == "magnitude":
                    # Remove smallest absolute weights
                    n_keep = max(1, int(w.numel() * keep_ratio))
                    flat = w.abs().flatten()
                    threshold_val = flat.kthvalue(max(1, n_keep)).values.item()
                    mask = w.abs() >= threshold_val
                    param.data.copy_(w * mask)

                elif method == "spectral":
                    # SVD: reconstruct with only top-k singular values
                    k_keep = max(1, int(k * keep_ratio))
                    try:
                        U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
                        # Keep only top k_keep singular values
                        S_k = S[:k_keep]
                        U_k = U[:, :k_keep]
                        Vh_k = Vh[:k_keep, :]
                        reconstructed = (U_k * S_k.unsqueeze(0)) @ Vh_k
                        param.data.copy_(reconstructed.half() if self.device == "cuda" else reconstructed)
                    except Exception as e:
                        print(f"    SVD failed on {name}: {e}")
                        continue

                elif method == "wanda":
                    # Wanda-style: weight × activation norm importance
                    # Simplified: use weight magnitude × row norm
                    row_norms = w.norm(dim=1, keepdim=True)
                    scores = w.abs() * row_norms
                    n_keep = max(1, int(w.numel() * keep_ratio))
                    threshold_val = scores.flatten().kthvalue(
                        max(1, n_keep)
                    ).values.item()
                    mask = scores >= threshold_val
                    param.data.copy_(w * mask)

                removed = (param == 0).sum().item() if method in ("magnitude", "wanda") else \
                    (w.numel() - k_keep * (m + n)) if method == "spectral" else 0
                total_removed += removed

                if name.endswith("weight"):
                    short_name = ".".join(name.split(".")[-3:])
                    stats[short_name] = {
                        "shape": [m, n],
                        "removed": removed,
                        "kept": param.numel() - removed,
                    }

        # Compute compression metrics
        pruned_params = original_params - total_removed
        param_bytes = 2 if self.device == "cuda" else 4  # half vs float

        result = {
            "model": self.model_id,
            "method": method,
            "original_m": round(original_params / 1e6, 1),
            "pruned_m": round(pruned_params / 1e6, 1),
            "original_gb": round(original_params * param_bytes / 1e9, 2),
            "pruned_gb": round(pruned_params * param_bytes / 1e9, 2),
            "compression_ratio": round(original_params / max(pruned_params, 1), 2),
            "removed_pct": round(100 * total_removed / original_params, 1),
            "per_layer": stats,
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            result["saved_to"] = save_path

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result
