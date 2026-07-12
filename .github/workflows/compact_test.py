#!/usr/bin/env python3
"""Quick test: 50% and 10% SVD compact, verify file size and loss delta."""
import os, sys, json, shutil, glob, time, torch
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.getcwd())
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from weightnoise.compress import factorize_weight, SVDLinear

HF_TOKEN = os.environ.get("HF_TOKEN", "")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen3.5-0.8B"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

log("Loading model...")
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN)
tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)

# Helper to test a compression ratio
def test_ratio(ratio):
    log(f"\n=== Testing keep_ratio={ratio} ===")
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN)
    stats = {"before": 0, "after": 0, "converted": 0, "skipped": 0}
    
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.ndim == 2:
            m, n = mod.weight.shape
            stats["before"] += m * n
            k_keep = max(1, int(min(m, n) * ratio))
            stored = k_keep * (m + n)
            if stored < m * n:
                U, S, Vt, k = factorize_weight(mod.weight.data, ratio)
                svd = SVDLinear(n, m, k, bias=mod.bias is not None)
                svd.U.data = U[:, :k]
                svd.S.data = S[:k]
                svd.Vt.data = Vt[:k, :]
                if mod.bias is not None:
                    svd.bias.data = mod.bias.data
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, svd)
                stats["converted"] += 1
                stats["after"] += k * (m + n)
            else:
                stats["after"] += m * n
                stats["skipped"] += 1
    
    model.eval()
    with torch.no_grad():
        texts = ["The future of AI depends on efficient models that run on local hardware."]
        ids = tok(texts[0], return_tensors="pt", truncation=True, max_length=64).input_ids
        loss = model(ids, labels=ids).loss.item()
    
    cr = stats["before"] / max(stats["after"], 1)
    params_saved = (stats["before"] - stats["after"]) / 1e6
    return {"ratio": ratio, "cr": round(cr, 3), "loss": round(loss, 4),
            "entries_before_m": round(stats["before"]/1e6, 1),
            "entries_after_m": round(stats["after"]/1e6, 1),
            "converted": stats["converted"], "skipped": stats["skipped"]}

# Test base loss first
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
texts = ["The future of AI depends on efficient models that run on local hardware."]
ids = tok(texts[0], return_tensors="pt", truncation=True, max_length=64).input_ids
with torch.no_grad():
    base_loss = model(ids, labels=ids).loss.item()
torch.cuda.empty_cache() if torch.cuda.is_available() else None
log(f"Base loss: {base_loss:.4f}")
del model

# Test at 50% and 10%
results = []
for r in [0.5, 0.1]:
    res = test_ratio(r)
    results.append(res)
    delta = res["loss"] - base_loss
    status = "✅" if abs(delta) < 0.5 else "❌"
    log(f"  {status} CR={res['cr']}x entries={res['entries_after_m']}M delta={delta:+.4f}")

# Print summary
print(f"\n{'='*50}", flush=True)
print(f"  SVD COMPACT TEST RESULTS", flush=True)
print(f"{'='*50}", flush=True)
for r in results:
    delta = r["loss"] - base_loss
    print(f"  keep={r['ratio']:.0%} | CR={r['cr']}x | {r['entries_before_m']}M->{r['entries_after_m']}M | loss={r['loss']:.4f} (delta={delta:+.4f})", flush=True)
print(f"  Time: {time.time()-t0:.0f}s", flush=True)

# If 50% is OK, push to HF
r50 = results[0]
if abs(r50["loss"] - base_loss) < 0.5 and r50["cr"] > 1.0:
    log("\n50% OK — pushing compressed model to HF...")
    # Re-create and save
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN)
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.weight.ndim == 2:
            m, n = mod.weight.shape
            k_keep = max(1, int(min(m, n) * 0.5))
            stored = k_keep * (m + n)
            if stored < m * n:
                U, S, Vt, k = factorize_weight(mod.weight.data, 0.5)
                svd = SVDLinear(n, m, k, bias=mod.bias is not None)
                svd.U.data = U[:, :k]; svd.S.data = S[:k]; svd.Vt.data = Vt[:k, :]
                if mod.bias is not None:
                    svd.bias.data = mod.bias.data
                parent_name, child_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
                setattr(parent, child_name, svd)
    
    model.save_pretrained("/tmp/compact_model")
    tok.save_pretrained("/tmp/compact_model")
    import json
    json.dump({"model_type": "qwen2"}, open("/tmp/compact_model/config.json", "w"))
    
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    api.upload_folder(folder_path="/tmp/compact_model", repo_id="KiriLabs/Qwen3.5-0.8B-SVD-Compact-1.5x", repo_type="model", ignore_patterns=[".*"])
    log("Pushed to HF: KiriLabs/Qwen3.5-0.8B-SVD-Compact-1.5x")
else:
    log(f"Skipping HF push — 50% result not acceptable (delta={abs(r50['loss']-base_loss):.4f})")
