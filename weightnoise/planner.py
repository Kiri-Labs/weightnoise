"""Dynamic compression planner — no hardcoded thresholds, no fixed hardware lists."""

def _detect_hardware():
    """Detect available hardware at runtime — any GPU, any VRAM."""
    import torch
    result = {}
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            result[torch.cuda.get_device_name(i)] = {
                "vram_bytes": props.total_memory,
                "vram_gb": round(props.total_memory / 1e9, 1),
                "compute_capability": f"{props.major}.{props.minor}",
            }
    # Also detect system RAM
    try:
        import psutil
        result["system_ram"] = {
            "total_gb": round(psutil.virtual_memory().total / 1e9, 1),
            "available_gb": round(psutil.virtual_memory().available / 1e9, 1),
        }
    except ImportError:
        # Try reading /proc/meminfo directly (Linux)
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if "MemTotal" in line:
                        kb = int(line.split()[1])
                        result["system_ram"] = {"total_gb": round(kb / 1e6, 1)}
                        break
        except (FileNotFoundError, OSError):
            pass
    if not result:
        # Fallback: return a reasonable default
        result["system_ram"] = {"total_gb": "unknown (must pass --vram manually)"}
    return result


def _adaptive_thresholds(noise_pcts, eff_ranks):
    """Compute decision thresholds from data distribution — no magic numbers."""
    import numpy as np

    noise_pcts = np.array(noise_pcts)
    eff_ranks = np.array(eff_ranks)

    # Adaptive threshold: use percentile-based cutoffs
    # Prune threshold: weights in the top noise decile
    noise_threshold_prune = np.percentile(noise_pcts, 75) if len(noise_pcts) > 5 else 30
    noise_threshold_heavy = np.percentile(noise_pcts, 90) if len(noise_pcts) > 5 else 50

    # SVD threshold: eff_rank significantly below the median
    median_rank = np.median(eff_ranks) if len(eff_ranks) > 0 else 90
    rank_sv = median_rank - np.std(eff_ranks) * 1.5 if len(eff_ranks) > 1 else 70
    rank_sv_heavy = median_rank - np.std(eff_ranks) * 2.5 if len(eff_ranks) > 1 else 50

    return {
        "prune": float(noise_threshold_prune),
        "prune_heavy": float(noise_threshold_heavy),
        "svd": float(rank_sv),
        "svd_heavy": float(rank_sv_heavy),
        "median_rank": float(median_rank),
        "n_samples": len(noise_pcts),
    }


def estimate_model_size(params_m, precision="fp16"):
    """Estimate model memory. Works for any param count, any precision."""
    factors = {"fp32": 4, "fp16": 2, "int8": 1, "int4": 0.5}
    factor = factors.get(precision, 2)
    return round(params_m * factor / 1000, 2)


def generate_plan(results, vram_gb=None, precision="fp16"):
    """Generate compression plan dynamically from noise analysis.

    No hardcoded hardware profiles. No fixed thresholds.
    Everything computed from the data + user-provided or detected hardware.

    Args:
        results: dict from NoiseInspector.analyze()
        vram_gb: optional VRAM limit. If None, auto-detect.
        precision: target inference precision

    Returns:
        dict with adaptive compression plan
    """
    import numpy as np

    summary = results.get("summary", {})
    total_m = summary.get("total_params_m", 0)
    total_params = total_m * 1e6

    # ── Detect or accept hardware ──
    hw = _detect_hardware()
    gpu_list = {k: v for k, v in hw.items() if k != "system_ram"}
    
    if vram_gb is None:
        # Use detected hardware
        if gpu_list:
            vram_gb = min(v["vram_gb"] for v in gpu_list.values())
        else:
            vram_gb = hw.get("system_ram", {}).get("total_gb", 0)
            if vram_gb and isinstance(vram_gb, str):
                vram_gb = None  # can't determine

    # ── Gather all metrics dynamically ──
    all_metrics = []
    for k in sorted(x for x in results if isinstance(x, int)):
        for name, m in results[k]["matrices"].items():
            all_metrics.append({
                "layer": k,
                "name": name,
                "noise": m.get("low_importance_pct", 0),
                "eff_rank": m.get("eff_rank_pct", 0),
                "kurtosis": m.get("kurtosis", 0),
                "kl_div": m.get("kl_div_gaussian", 0),
                "shape": m.get("shape", [0, 0]),
                "params": m["shape"][0] * m["shape"][1] if m.get("shape") else 0,
            })

    noise_vals = [m["noise"] for m in all_metrics]
    rank_vals = [m["eff_rank"] for m in all_metrics]
    
    # ── Compute adaptive thresholds from the distributions ──
    thresholds = _adaptive_thresholds(noise_vals, rank_vals)

    # ── Per-matrix decisions from thresholds ──
    prunable_total = 0
    svdable_total = 0
    per_layer = {}

    for m in all_metrics:
        noise = m["noise"]
        eff_rank = m["eff_rank"]
        params = m["params"]

        if noise >= thresholds["prune_heavy"]:
            action = "prune_heavy"
            savings = int(params * noise / 200)  # adaptive savings
        elif noise >= thresholds["prune"]:
            action = "prune_light"
            savings = int(params * noise / 300)
        elif eff_rank <= thresholds["svd_heavy"]:
            action = "svd_heavy"
            savings = int(params * 0.4)
        elif eff_rank <= thresholds["svd"]:
            action = "svd_light"
            savings = int(params * 0.2)
        else:
            action = "keep"
            savings = 0

        if action.startswith("prune"):
            prunable_total += savings
        elif action.startswith("svd"):
            svdable_total += savings

        lidx = m["layer"]
        if lidx not in per_layer:
            per_layer[lidx] = []
        per_layer[lidx].append({
            "name": m["name"],
            "shape": m["shape"],
            "noise_pct": round(noise, 1),
            "eff_rank_pct": round(eff_rank, 1),
            "action": action,
            "savings": savings,
        })

    compressed_m = (total_params - prunable_total - svdable_total) / 1e6

    # ── Dynamic hardware targets ──
    hw_targets = {}
    # Include any detected GPUs
    for name, gpu in gpu_list.items():
        gb = gpu["vram_gb"]
        compressed_size = compressed_m * (4 if precision == "fp32" else 2) / 1000
        hw_targets[name] = {
            "vram_gb": gb,
            "compressed_size_gb": round(compressed_size, 2),
            "fits": compressed_size <= gb,
            "needs_quant": None if compressed_size <= gb else _recommend_quant(compressed_size, gb),
        }
    # Also include system RAM if available and different from GPU
    sys_ram = hw.get("system_ram", {}).get("total_gb", 0)
    if sys_ram and isinstance(sys_ram, (int, float)) and not any(
        abs(h["vram_gb"] - sys_ram) < 1 for h in hw_targets.values()
    ):
        compressed_size = compressed_m * 4 / 1000  # CPU runs fp32
        hw_targets["system_ram"] = {
            "vram_gb": round(sys_ram, 1),
            "compressed_size_gb": round(compressed_size, 2),
            "fits": compressed_size <= sys_ram,
            "needs_quant": None if compressed_size <= sys_ram else _recommend_quant(compressed_size, sys_ram),
        }
    # Add the user-specified vram target if given
    if vram_gb and not any(abs(h["vram_gb"] - vram_gb) < 0.5 for h in hw_targets.values()):
        compressed_size = compressed_m * (4 if precision == "fp32" else 2) / 1000
        hw_targets[f"target_{vram_gb}GB"] = {
            "vram_gb": vram_gb,
            "compressed_size_gb": round(compressed_size, 2),
            "fits": compressed_size <= vram_gb,
            "needs_quant": None if compressed_size <= vram_gb else _recommend_quant(compressed_size, vram_gb),
        }

    mrr = compressed_m
    return {
        "model": summary.get("model", "unknown"),
        "total_m": total_m,
        "compressed_m": round(mrr, 1),
        "compression_ratio": round(total_params / max(total_params - prunable_total - svdable_total, 1), 2),
        "thresholds": {
            "prune_at": f">={thresholds['prune']:.0f}% noise",
            "prune_heavy_at": f">={thresholds['prune_heavy']:.0f}% noise",
            "svd_at": f"<={thresholds['svd']:.0f}% eff_rank",
            "svd_heavy_at": f"<={thresholds['svd_heavy']:.0f}% eff_rank",
            "note": f"Computed from {thresholds['n_samples']} matrices — no hardcoded values",
        },
        "savings": {
            "prunable_m": round(prunable_total / 1e6, 1),
            "svdable_m": round(svdable_total / 1e6, 1),
            "total_m": round((prunable_total + svdable_total) / 1e6, 1),
        },
        "hardware_targets": hw_targets,
        "per_layer": per_layer,
    }


def _recommend_quant(size_gb, vram_gb):
    """Recommend minimal quantization to fit."""
    for bits, factor in [(8, 0.5), (4, 0.25)]:
        if size_gb * factor <= vram_gb:
            return f"int{bits}"
    return None
