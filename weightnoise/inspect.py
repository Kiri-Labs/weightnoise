"""Core inspection engine: loads model, scans weights, computes noise metrics.

Supports multiple layer naming conventions:
  - 'layers.N.' (Qwen, LLaMA, Mistral)
  - 'transformer.h.N.' (GPT-2, distilgpt2)
  - 'model.layers.N.' (some Qwen variants)
"""
import torch
import numpy as np
import re
import time
from collections import defaultdict
from transformers import AutoModelForCausalLM


# No hardcoded layer patterns. Dynamically discovered from model parameters.


class NoiseInspector:
    """Analyze neural network weights for noise content.

    Metrics per weight matrix:
      - Distribution shape (mean, std, skew, kurtosis)
      - Gaussian fit divergence (KL from ideal Gaussian of same params)
      - Singular value spectrum (via SVD)
      - Effective rank (fraction of rank needed for 99% energy)
      - Concentration ratio (energy in top-k vs total)
    """

    def __init__(self, model_id: str, device: str = "cpu", trust_remote_code: bool = False):
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code

    def _find_layer_pattern(self, param_names):
        """Dynamically discover layer numbering pattern from parameter names.
        
        No hardcoded patterns. Finds any 'prefix.N.suffix' where N is a layer index
        and the prefix appears repeatedly in the model.
        """
        # Find all parameters with a digit run that could be a layer index
        candidates = {}
        for name in param_names:
            m = re.search(r'^(.*[^\d])(\d+)([._].*)', name)
            if m:
                prefix, num_str, suffix = m.groups()
                # Only consider 1-3 digit numbers that appear with same prefix
                if 1 <= len(num_str) <= 3:
                    key = (prefix, suffix)
                    candidates.setdefault(key, set()).add(int(num_str))
        
        # Find the pattern that spans the most contiguous layers
        best = None
        best_span = 0
        for (prefix, suffix), layers in candidates.items():
            if len(layers) > 1:
                span = max(layers) - min(layers)
                if span > best_span:
                    best_span = span
                    best = re.compile(re.escape(prefix) + r'(\d+)' + re.escape(suffix))
        
        if best:
            return best
        
        # Last resort: any '\d+' in 2D parameter names with '.weight' suffix
        for name in param_names:
            if '.weight' in name:
                m = re.search(r'(\d+)', name)
                if m:
                    idx = m.start(1)
                    prefix = name[:idx]
                    suffix = name[idx + len(m.group(1)):]
                    return re.compile(re.escape(prefix) + r'(\d+)' + re.escape(suffix))
        
        return re.compile(r'(\d+)')

    def analyze(self, threshold=None):
        """Run full noise analysis on the model.

        Args:
            threshold: Override for the per-row noise floor.
                       If None, computed adaptively from weight distribution.
                       If set (e.g. 0.01), scores below this fraction of row
                       max are classified as noise.

        Returns:
            dict with per-layer results and summary
        """
        import warnings
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )

        param_names = [n for n, _ in model.named_parameters()]
        layer_pattern = self._find_layer_pattern(param_names)

        # Group weight matrices by layer
        layer_groups = defaultdict(list)
        for name, param in model.named_parameters():
            match = layer_pattern.search(name)
            if match and param.ndim == 2:
                layer_groups[int(match.group(1))].append((name, param))

        if not layer_groups:
            # No layer structure — treat non-embedding 2D params as single group
            for name, param in model.named_parameters():
                if param.ndim == 2 and 'embed' not in name.lower() and 'lm_head' not in name.lower():
                    layer_groups[0].append((name, param))

        total_params = sum(p.numel() for p in model.parameters())
        total_noise_params = 0
        results = {}

        for layer_idx in sorted(layer_groups.keys()):
            mats = layer_groups[layer_idx]
            layer_result = {"matrices": {}, "noise_params": 0, "total_params": 0}

            for name, param in mats:
                if param.ndim != 2:
                    continue
                w = param.detach().float().cpu().numpy()
                layer_result["total_params"] += w.size

                # === Distribution metrics ===
                mean = float(np.mean(w))
                std = float(np.std(w))
                if std > 0:
                    skew = float(np.mean(((w - mean) / std) ** 3))
                    kurt = float(np.mean(((w - mean) / std) ** 4)) - 3.0
                else:
                    skew = 0.0
                    kurt = -2.0

                # === KL divergence from best-fit Gaussian ===
                # Measures how much the weight distribution differs from random noise
                hist, bin_edges = np.histogram(w.flatten(), bins=min(100, w.size // 10 + 1), density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                gauss = (1.0 / (std * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((bin_centers - mean) / (std + 1e-12)) ** 2)
                gauss = gauss / (gauss.sum() + 1e-12)
                hist_norm = hist / (hist.sum() + 1e-12)
                hist_norm = np.clip(hist_norm, 1e-12, None)
                gauss = np.clip(gauss, 1e-12, None)
                kl_div = float(np.sum(hist_norm * np.log(hist_norm / gauss)))

                # === SVD-based noise analysis ===
                m, n = w.shape
                # Use randomized SVD for speed (cap at 512 SV)
                k_max = min(m, n, 512)
                try:
                    # Use scipy or numpy for SVD
                    if k_max >= min(m, n):
                        U, s, Vh = np.linalg.svd(w, full_matrices=False)
                    else:
                        # Truncated via randomized SVD
                        U, s, Vh = self._randomized_svd(w, k_max)
                except Exception:
                    s = np.array([0.0])

                # === Key noise metrics ===
                if len(s) > 0 and s[0] > 0:
                    # 1. Spectral noise ratio: singular values below noise floor
                    noise_floor = threshold * s[0]
                    noise_svs = (s < noise_floor).sum()
                    noise_sv_pct = 100.0 * noise_svs / len(s)

                    # 2. Energy concentration
                    total_energy = float(np.sum(s ** 2))
                    noise_energy = float(np.sum(s[s < noise_floor] ** 2))
                    noise_energy_pct = 100.0 * noise_energy / (total_energy + 1e-12)

                    # 3. Effective rank at 90%, 95%, 99% energy
                    cum_energy = np.cumsum(s ** 2) / (total_energy + 1e-12)
                    rank_90 = int(np.searchsorted(cum_energy, 0.90) + 1)
                    rank_95 = int(np.searchsorted(cum_energy, 0.95) + 1)
                    rank_99 = int(np.searchsorted(cum_energy, 0.99) + 1)
                    total_rank = len(s)

                    # 4. Concentration of top 10% of singular values
                    top10 = max(1, total_rank // 10)
                    top10_energy = float(np.sum(s[:top10] ** 2))
                    top10_concentration = 100.0 * top10_energy / (total_energy + 1e-12)

                    # 5. Effective rank (Renyi) — continuous measure
                    p = s / (s.sum() + 1e-12)
                    entropy = -np.sum(p * np.log(p + 1e-12))
                    eff_rank = np.exp(entropy)
                    eff_rank_pct = 100.0 * eff_rank / total_rank

                    # 6. Adaptive magnitude threshold: 2*MAD (median absolute deviation)
                    # = robust to outliers, no hardcoded sigma
                    median_val = np.median(np.abs(w))
                    mad = np.median(np.abs(np.abs(w) - median_val))
                    mag_threshold = median_val + 2 * mad if mad > 0 else 0.1 * std
                    near_zero_pct = 100.0 * (np.abs(w) < mag_threshold).mean()

                    # 7. Weight scores based on Wanda-like metric (|w| × column norm)
                    col_norms = np.linalg.norm(w, axis=0, keepdims=True)
                    importance_scores = np.abs(w) * col_norms
                    row_max = importance_scores.max(axis=1, keepdims=True)
                    valid_rows = row_max[:, 0] > 1e-12
                    if valid_rows.any():
                        relative_scores = importance_scores / np.where(row_max > 1e-12, row_max, 1.0)
                        # Adaptive noise threshold: if None, compute from data
                        # Use the median relative score × 2 as the noise floor
                        if threshold is None:
                            all_relative = relative_scores[valid_rows].flatten()
                            adaptive_thresh = float(np.percentile(all_relative, 5))
                            noise_i = (relative_scores < max(adaptive_thresh, 1e-6)).astype(float)
                        else:
                            noise_i = (relative_scores < threshold).astype(float)
                        low_importance_pct = 100.0 * noise_i.mean()
                    else:
                        low_importance_pct = 0.0
                else:
                    noise_sv_pct = 0.0
                    noise_energy_pct = 0.0
                    rank_90 = 0
                    rank_95 = 0
                    rank_99 = 0
                    total_rank = k_max
                    top10_concentration = 0.0
                    eff_rank_pct = 100.0
                    near_zero_pct = 0.0

                mat_result = {
                    "shape": [m, n],
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "skew": round(skew, 4),
                    "kurtosis": round(kurt, 4),
                    "kl_div_gaussian": round(kl_div, 4),
                    "noise_sv_pct": round(noise_sv_pct, 1),
                    "noise_energy_pct": round(noise_energy_pct, 1),
                    "rank_90_pct": round(100.0 * rank_90 / max(total_rank, 1), 1),
                    "rank_95_pct": round(100.0 * rank_95 / max(total_rank, 1), 1),
                    "rank_99_pct": round(100.0 * rank_99 / max(total_rank, 1), 1),
                    "top10_concentration": round(top10_concentration, 1),
                    "eff_rank_pct": round(eff_rank_pct, 1),
                    "near_zero_pct": round(near_zero_pct, 1),
                    "low_importance_pct": round(low_importance_pct, 1),
                    "param_count": w.size,
                }
                layer_result["matrices"][name] = mat_result

                # Estimate noise params: weights with low importance
                noise_params = int(w.size * low_importance_pct / 100.0)
                layer_result["noise_params"] += noise_params
                total_noise_params += noise_params

            results[layer_idx] = layer_result

        # Summary
        signal_params = max(1, total_params - total_noise_params)
        results["summary"] = {
            "model": self.model_id,
            "total_params_m": round(total_params / 1e6, 1),
            "noise_percentage": round(100.0 * total_noise_params / max(total_params, 1), 1),
            "estimated_signal_m": round(signal_params / 1e6, 1),
            "estimated_noise_m": round(total_noise_params / 1e6, 1),
            "num_layers": len(layer_groups),
            "num_matrices": sum(len(g["matrices"]) for g in results.values()),
            "analysis_threshold": threshold if threshold is not None else "adaptive (5th percentile)",
            "threshold_note": "Threshold auto-computed from data distribution. Pass --threshold to override." if threshold is None else None,
        }

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def _randomized_svd(self, M, k):
        """Randomized SVD for fast truncated decomposition.
        
        Uses numpy.linalg.svd on a smaller random projection of the matrix.
        """
        m, n = M.shape
        rng = np.random.RandomState(42)
        
        # Step 1: Random projection
        omega = rng.randn(n, k + 10).astype(M.dtype)
        Q = M @ omega
        Q, _ = np.linalg.qr(Q)
        
        # Step 2: Project and SVD the smaller matrix
        B = Q.T @ M
        Uhat, s, Vt = np.linalg.svd(B, full_matrices=False)
        U = Q @ Uhat
        
        return U, s, Vt
