"""Compression planner: from noise analysis → actionable compression plan."""
import json

# Hardware profiles for local deployment
HARDWARE_PROFILES = {
    "rtx_4090": {"vram_gb": 24, "name": "RTX 4090"},
    "rtx_3090": {"vram_gb": 24, "name": "RTX 3090"},
    "rtx_4080": {"vram_gb": 16, "name": "RTX 4080"},
    "rtx_3080": {"vram_gb": 12, "name": "RTX 3080"},
    "mac_m1": {"vram_gb": 8, "name": "Apple M1 (8GB)"},
    "mac_m2": {"vram_gb": 24, "name": "Apple M2 Max (24GB)"},
    "colab_free": {"vram_gb": 15, "name": "Colab T4 Free (15GB)"},
    "colab_pro": {"vram_gb": 40, "name": "Colab A100 (40GB)"},
}


def estimate_model_size(model_config):
    """Estimate model memory requirements."""
    if isinstance(model_config, dict):
        params_m = model_config.get("total_params_m", 0)
    else:
        params_m = model_config
    return {
        "fp32_gb": round(params_m * 4 / 1000, 1),
        "fp16_gb": round(params_m * 2 / 1000, 1),
        "int8_gb": round(params_m * 1 / 1000, 1),
        "int4_gb": round(params_m * 0.5 / 1000, 2),
    }


def generate_plan(results):
    """Generate a compression plan from noise analysis results.
    
    Args:
        results: dict from NoiseInspector.analyze()
    
    Returns:
        dict with compression plan and hardware recommendations
    """
    summary = results.get("summary", {})
    total_m = summary.get("total_params_m", 0)
    noise_pct = summary.get("noise_percentage", 0)
    noise_m = summary.get("estimated_noise_m", 0)
    
    # Estimate sizes
    sizes = estimate_model_size(total_m)
    
    # Per-layer plan
    layer_plans = {}
    total_prunable = 0
    total_svdable = 0
    
    for k in sorted(x for x in results if isinstance(x, int)):
        layer = results[k]
        mats = layer["matrices"]
        
        plan = []
        for name, m in sorted(mats.items()):
            noise = m.get("low_importance_pct", 0)
            eff_rank = m.get("eff_rank_pct", 0)
            shape = m.get("shape", [0, 0])
            params = shape[0] * shape[1]
            
            # Decision logic based on metrics
            if noise > 30:
                action = "prune_heavy"
                savings = int(params * 0.3)
            elif noise > 15:
                action = "prune_light"
                savings = int(params * 0.15)
            elif eff_rank < 70:
                action = "svd_truncate"
                savings = int(params * 0.3)
            elif eff_rank < 85:
                action = "svd_light"
                savings = int(params * 0.15)
            else:
                action = "keep"
                savings = 0
            
            if action.startswith("prune"):
                total_prunable += savings
            elif action.startswith("svd"):
                total_svdable += savings
            
            plan.append({
                "name": name,
                "shape": shape,
                "params": params,
                "noise_pct": noise,
                "eff_rank_pct": eff_rank,
                "action": action,
                "savings": savings,
            })
        
        layer_plans[k] = plan
    
    # Total achievable compression
    total_params = total_m * 1e6
    total_savings = total_prunable + total_svdable
    compressed_m = (total_params - total_savings) / 1e6
    
    # Hardware targets
    hardware_targets = {}
    for key, hw in HARDWARE_PROFILES.items():
        # fp16 is the standard for inference
        fp16_size = total_m * 2 / 1000  # GB at fp16
        compressed_size = compressed_m * 2 / 1000  # GB after compression
        
        # Can it fit?
        needs_quant = compressed_size > hw["vram_gb"]
        quant_level = None
        if needs_quant:
            for bits, factor in [(8, 1), (4, 0.5)]:
                if compressed_m * factor / 1000 <= hw["vram_gb"]:
                    quant_level = f"int{bits}"
                    break
            else:
                quant_level = "too_large"
        
        hardware_targets[key] = {
            "name": hw["name"],
            "vram_gb": hw["vram_gb"],
            "fits_fp16": compressed_size <= hw["vram_gb"],
            "fits_fp32": sizes["fp32_gb"] <= hw["vram_gb"],
            "needs_quant": quant_level,
            "compressed_size_gb": round(compressed_size, 2),
        }
    
    return {
        "model": summary.get("model", "unknown"),
        "total_m": total_m,
        "noise_m": noise_m,
        "noise_pct": noise_pct,
        "uncompressed_sizes": sizes,
        "plan": {
            "prunable_params_m": round(total_prunable / 1e6, 1),
            "svdable_params_m": round(total_svdable / 1e6, 1),
            "compressed_m": round(compressed_m, 1),
            "compression_ratio": round(total_params / max(total_params - total_savings, 1), 2),
            "layer_count": len(layer_plans),
        },
        "per_layer": layer_plans,
        "hardware_targets": hardware_targets,
        "recommendation": _recommend(hardware_targets, noise_pct, compressed_m),
    }


def _recommend(hardware, noise_pct, compressed_m):
    """Generate a plain-English recommendation."""
    # Find best fit
    for key in ["colab_free", "rtx_4090", "mac_m2", "rtx_4080", "rtx_3080"]:
        hw = hardware.get(key)
        if hw and not hw.get("needs_quant"):
            return (
                f"Fits on {hw['name']} ({hw['vram_gb']}GB) at fp16 "
                f"with {round(compressed_m, 0)}M params uncompressed. "
                f"{'Prune ' + str(round(noise_pct)) + '% noise first.' if noise_pct > 10 else 'No pruning needed.'}"
            )
    
    # Needs quantization
    for key in ["rtx_4090", "rtx_4080", "colab_free"]:
        hw = hardware.get(key)
        if hw and hw.get("needs_quant") and hw["needs_quant"] != "too_large":
            return (
                f"Needs {hw['needs_quant']} quantization to fit on {hw['name']} "
                f"({hw['vram_gb']}GB). {round(compressed_m, 0)}M params compressed "
                f"to ~{round(compressed_m * (1 if hw['needs_quant']=='int8' else 0.5), 0)}M."
            )
    
    return "Model too large for consumer hardware even after compression."
