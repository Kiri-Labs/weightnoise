"""Core inspection engine: loads model, scans weights, computes noise metrics."""
import torch
import numpy as np
import re
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoConfig


class NoiseInspector:
    """Analyze neural network weights for noise content.

    Metrics per weight matrix:
      - Distribution shape (mean, std, skew, kurtosis)
      - Signal-to-noise ratio (|mean| / std)
      - Gaussan fit divergence (KL from ideal Gaussian)
      - Singular value spectrum (via SVD)
      - Spectral noise floor (small singular values)
    """

    def __init__(self, model_id: str, device: str = "cpu", trust_remote_code: bool = False):
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code

    def analyze(self, threshold: float = 0.01):
        """Run full noise analysis on the model.

        Args:
            threshold: Fraction of max singular value below which = noise.
                       Default 0.01 means singular values <1% of max are noise.

        Returns:
            dict with per-layer results and summary
        """
        print(f"  Loading {self.model_id}...")
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )
        print(f"  Scanning {sum(1 for _ in model.named_parameters())} parameters...")

        # Group weight matrices by layer
        layer_groups = defaultdict(list)
        non_layer_params = {}
        total_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            match = re.search(r'layers\.(\d+)', name)
            if match and param.ndim == 2:
                layer_idx = int(match.group(1))
                layer_groups[layer_idx].append((name, param))
            elif param.ndim == 2:
                non_layer_params[name] = param

        if not layer_groups:
            # Model doesn't use 'layers' naming — treat everything as ungrouped
            for name, param in model.named_parameters():
                if param.ndim == 2:
                    match = re.search(r'(\d+)', name)
                    idx = int(match.group(1)) if match else 0
                    layer_groups[idx].append((name, param))

        results = {}
        total_noise_params = 0
        total_signal_params = 0
        total_compressible = 0

        print(f"  Found {len(layer_groups)} layer groups")

        for layer_idx in sorted(layer_groups.keys()):
            mats = layer_groups[layer_idx]
            layer_result = {"matrices": {}, "noise_params": 0, "total_params": 0}

            for name, param in mats:
                # Ensure 2D
                if param.ndim != 2:
                    continue

                w = param.float().cpu().numpy()
                layer_result["total_params"] += w.size

                # === Distribution metrics ===
                mean = float(np.mean(w))
                std = float(np.std(w))
                if std > 0:
                    skew = float(np.mean(((w - mean) / std) ** 3))
                    kurt = float(np.mean(((w - mean) / std) ** 4)) - 3  # excess kurtosis
                else:
                    skew = 0.0
                    kurt = -2.0

                # Gaussian noise fit: how close is this to a Gaussian?
                # KL divergence between actual distribution and best-fit Gaussian
                hist, bin_edges = np.histogram(w.flatten(), bins=50, density=True)
                bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                gaussian_pdf = (1 / (std * np.sqrt(2 * np.pi))) * \
                    np.exp(-0.5 * ((bin_centers - mean) / std) ** 2)
                gaussian_pdf = gaussian_pdf / gaussian_pdf.sum()
                hist_norm = hist / hist.sum()
                # Avoid log(0)
                hist_norm = np.clip(hist_norm, 1e-12, None)
                gaussian_pdf = np.clip(gaussian_pdf, 1e-12, None)
                kl_div = float(np.sum(hist_norm * np.log(hist_norm / gaussian_pdf)))

                # SNR: |mean| / std — high SNR = structured, low SNR = noise-like
                snr = abs(mean) / (std + 1e-12)

                # === Spectral noise analysis ===
                m, n = w.shape
                k = min(m, n, 200)  # Cap at 200 singular values for speed
                try:
                    _, s, _ = np.linalg.svd(w, full_matrices=False)
                    s = s[:k]
                except:
                    s = np.array([0.0])

                # Noise floor: singular values below threshold * max(s)
                if len(s) > 0 and s[0] > 0:
                    noise_floor = threshold * s[0]
                    noise_svals = (s < noise_floor).sum()
                    noise_ratio = float(noise_svals / len(s))

                    # Energy in noise vs signal
                    total_energy = float(np.sum(s ** 2))
                    noise_energy = float(np.sum(s[s < noise_floor] ** 2))
                    signal_energy = total_energy - noise_energy
                    noise_energy_pct = 100 * noise_energy / (total_energy + 1e-12)

                    # Compressibility estimate: rank needed to retain 99% energy
                    cum_energy = np.cumsum(s ** 2) / (total_energy + 1e-12)
                    rank_99 = int(np.searchsorted(cum_energy, 0.99) + 1)
                    compressibility = 1.0 - rank_99 / len(s)

                    # Fraction of weights near zero (within 2 std of noise floor)
                    noise_weights_pct = 100.0 * (
                        np.abs(w.flatten()) < mean + 2 * std
                    ).mean()
                else:
                    noise_ratio = 0.0
                    noise_energy_pct = 0.0
                    compressibility = 0.0
                    noise_weights_pct = 0.0

                # === Magnitude-based noise: weights below noise threshold ===
                weight_threshold = std * threshold * 100  # Scale threshold by std
                small_weights = (np.abs(w) < weight_threshold).mean() * 100

                mat_result = {
                    "shape": list(w.shape),
                    "mean": round(mean, 6),
                    "std": round(std, 6),
                    "skew": round(skew, 4),
                    "kurtosis": round(kurt, 4),
                    "snr": round(snr, 4),
                    "kl_div_gaussian": round(kl_div, 4),
                    "noise_svals_ratio": round(100 * noise_ratio, 1),
                    "noise_energy_pct": round(noise_energy_pct, 1),
                    "compressibility_99pct": round(100 * compressibility, 1),
                    "weights_near_zero_pct": round(noise_weights_pct, 1),
                    "small_weights_pct": round(small_weights, 1),
                    "param_count": w.size,
                }
                layer_result["matrices"][name] = mat_result

                # Classify "noise" weights (small magnitude relative to distribution)
                noise_count = int(w.size * small_weights / 100)
                layer_result["noise_params"] += noise_count
                total_noise_params += noise_count
                total_signal_params += w.size - noise_count
                total_compressible += int(w.size * compressibility)

            results[layer_idx] = layer_result

        # Summary
        summary = {
            "model": self.model_id,
            "total_params_m": round(total_params / 1e6, 1),
            "noise_percentage": round(100 * total_noise_params / max(total_params, 1), 1),
            "compressible_percentage": round(100 * total_compressible / max(total_params, 1), 1),
            "estimated_signal_m": round(total_signal_params / 1e6, 1),
            "estimated_noise_m": round(total_noise_params / 1e6, 1),
            "num_layers": len(layer_groups),
            "num_matrices": sum(len(g["matrices"]) for g in results.values()),
            "analysis_threshold": threshold,
            "method_used": "spectral_noise_floor",
        }
        results["summary"] = summary

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return results

    def analyze_layer(self, layer_idx: int, threshold: float = 0.01):
        """Quick single-layer analysis."""
        results = self.analyze(threshold=threshold)
        if layer_idx in results:
            return {layer_idx: results[layer_idx], "summary": results["summary"]}
        return {"error": f"Layer {layer_idx} not found"}
