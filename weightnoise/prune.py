"""Pruning engine: remove noise from model weights.

Implements three methods:
  - magnitude: Zero out weights with smallest absolute values (baseline)
  - spectral: SVD truncation — keep only top-k singular values
  - wanda: Weight x activation norm importance (needs calibration data)
"""
import torch
import numpy as np
import os
from transformers import AutoModelForCausalLM


class NoisePruner:
    """Remove noise from model weights."""

    def __init__(self, model_id: str, device: str = "cpu", trust_remote_code: bool = False):
        self.model_id = model_id
        self.device = device
        self.trust_remote_code = trust_remote_code

    def _prepare_calibration_data(self, model, n_samples: int = 64, seq_len: int = 128):
        """Gather calibration data for Wanda-style scoring.

        Runs a few forward passes to measure activation norms.
        """
        import torch.utils.data
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=self.trust_remote_code)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

        # Use the model's own embedding as surrogate calibration
        # Generate random tokens (uniform over vocab)
        vocab_size = model.config.vocab_size
        input_ids = torch.randint(0, min(vocab_size, 50000), (n_samples, seq_len), device=self.device)
        attention_mask = torch.ones_like(input_ids)

        return input_ids, attention_mask

    def prune(self, method: str = "magnitude", keep_ratio: float = 0.5,
              threshold: float = 0.01, save_path: str = None):
        """Prune noise from model weights.

        Args:
            method: 'magnitude' — zero out smallest weights
                    'spectral' — SVD truncation
                    'wanda' — weight x activation norm scoring
            keep_ratio: Fraction of weights to keep (0.5 = keep 50%)
            threshold: Noise threshold for spectral method
            save_path: Path to save pruned model

        Returns:
            dict with pruning stats
        """
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            dtype=torch.float32,
            device_map=self.device,
            trust_remote_code=self.trust_remote_code,
        )

        original_params = sum(p.numel() for p in model.parameters())
        total_removed = 0
        stats = {}

        if method == "wanda":
            # Measure activation norms
            input_ids, attention_mask = self._prepare_calibration_data(model)
            activations = {}

            # Register hooks to capture input activations
            def get_act_hook(name):
                def hook(module, inp, out):
                    activations[name] = inp[0].detach()
                return hook

            handles = []
            for name, module in model.named_modules():
                if any(x in name for x in ["attn", "mlp", "dense", "fc"]):
                    handles.append(module.register_forward_hook(get_act_hook(name)))

            # Forward pass
            with torch.no_grad():
                model(input_ids, attention_mask=attention_mask)

            for h in handles:
                h.remove()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.ndim != 2:
                    continue

                w = param.float()
                m, n = w.shape

                if method == "magnitude":
                    n_keep = max(1, int(w.numel() * keep_ratio))
                    pos = max(1, w.numel() - n_keep)
                    flat = w.abs().flatten()
                    threshold_val = flat.kthvalue(pos).values.item()
                    mask = w.abs() >= threshold_val
                    param.data.copy_(w * mask)

                elif method == "spectral":
                    k_keep = max(1, int(min(m, n) * keep_ratio))
                    try:
                        U, S, Vh = torch.linalg.svd(w.float(), full_matrices=False)
                        U_k = U[:, :k_keep]
                        S_k = S[:k_keep]
                        Vh_k = Vh[:k_keep, :]
                        reconstructed = (U_k * S_k.unsqueeze(0)) @ Vh_k
                        param.data.copy_(reconstructed)
                    except Exception:
                        # Fallback to magnitude
                        flat = w.abs().flatten()
                        n_keep = max(1, int(w.numel() * keep_ratio))
                        pos = max(1, w.numel() - n_keep)
                        thresh = flat.kthvalue(pos).values.item()
                        param.data.copy_(w * (w.abs() >= thresh))

                elif method == "wanda":
                    if name in activations:
                        act_norm = activations[name].norm(dim=-1, keepdim=True)
                    else:
                        act_norm = torch.ones((1, 1), device=w.device)

                    scores = w.abs() * act_norm
                    n_keep = max(1, int(w.numel() * keep_ratio))
                    pos = max(1, w.numel() - n_keep)
                    thresh_val = scores.flatten().kthvalue(pos).values.item()
                    mask = scores >= thresh_val
                    param.data.copy_(w * mask)

                removed = (param.data == 0).sum().item() if method in ("magnitude", "wanda") else 0
                total_removed += removed if method in ("magnitude", "wanda") else \
                    w.numel() - k_keep * (m + n)

                short_name = ".".join(name.split(".")[-3:])
                stats[short_name] = {
                    "shape": [m, n],
                    "removed": removed if method in ("magnitude", "wanda") else w.numel() - k_keep * (m + n),
                }

        pruned_params = original_params - total_removed

        result = {
            "model": self.model_id,
            "method": method,
            "keep_ratio": keep_ratio,
            "original_m": round(original_params / 1e6, 1),
            "pruned_m": round(pruned_params / 1e6, 1),
            "original_gb": round(original_params * 4 / 1e9, 2),
            "pruned_gb": round(pruned_params * 4 / 1e9, 2),
            "compression_ratio": round(original_params / max(pruned_params, 1), 2),
            "removed_pct": round(100.0 * total_removed / original_params, 1),
        }

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            model.save_pretrained(save_path)
            result["saved_to"] = save_path

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return result

    def evaluate_perplexity(self, model, max_samples: int = 5):
        """Quick perplexity evaluation on a small test set."""
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or "<|endoftext|>"

        # Use a tiny calibration text — short phrases
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming how we process information.",
            "In the beginning, there was data and algorithms.",
            "The future of artificial intelligence depends on efficient models.",
            "Language models understand context through attention mechanisms.",
        ][:max_samples]

        total_loss = 0.0
        total_tokens = 0

        for text in texts:
            inputs = tokenizer(text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                total_loss += outputs.loss.item() * inputs["input_ids"].shape[1]
                total_tokens += inputs["input_ids"].shape[1]

        return np.exp(total_loss / max(total_tokens, 1))
