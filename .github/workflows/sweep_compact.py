#!/usr/bin/env python3
"""Sweep extreme SVD compression ratios on Qwen3.5-0.8B.

Tests keep_ratios: 50%, 20%, 10%, 5%, 2% to find the breaking point.
Measures: file size, param count, perplexity delta, reasoning quality.
"""
import os, sys, torch, shutil, json, glob, time
sys.stdout.reconfigure(line_buffering=True)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Allow importing weightnoise from source
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from weightnoise.compress import compress_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL = "Qwen/Qwen3.5-0.8B"
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

log("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
base.eval()

# Reference: save original size
original_dir = "/tmp/orig_qwen"
base.save_pretrained(original_dir, safe_serialization=False)
orig_size = sum(os.path.getsize(f) for f in glob.glob(f"{original_dir}/*.bin"))
orig_entries = sum(p.numel() for p in base.parameters())
orig_mb = orig_size / (1024*1024)
shutil.rmtree(original_dir)

short_texts = [
    "The future of AI depends on efficient models that run on local hardware.",
    "Neural networks learn hierarchical representations from data.",
    "Weight-space intelligence transfer enables cross-architecture knowledge.",
    "Quantum computing may revolutionize machine learning in the coming decades.",
    "Large language models require significant computational resources."
]

def avg_loss(model, tok, texts, max_len=128):
    losses = []
    with torch.no_grad():
        for txt in texts:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).input_ids
            losses.append(model(ids, labels=ids).loss.item())
    return sum(losses)/len(losses) if losses else 0.0

base_loss = avg_loss(base, tok, short_texts)
log(f"Base loss: {base_loss:.4f}")

# === SWEEP ===
ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
results = []

for ratio in ratios:
    log(f"\n=== Testing keep_ratio={ratio} ===")
    try:
        # Reload base model for each run
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
        compressed, stats = compress_model(model, keep_ratio=ratio, method="svd-observe")
        
        # Save and measure
        out = f"/tmp/compact_{ratio}"
        compressed.save_pretrained(out, safe_serialization=False)
        comp_size = sum(os.path.getsize(f) for f in glob.glob(f"{out}/*.bin"))
        comp_mb = comp_size / (1024*1024)
        comp_entries = sum(p.numel() for p in compressed.parameters())
        
        # Reload for eval
        loaded = AutoModelForCausalLM.from_pretrained(out, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
        loaded.eval()
        comp_loss = avg_loss(loaded, tok, short_texts)
        delta = comp_loss - base_loss
        
        # Cleanup
        shutil.rmtree(out)
        
        result = {
            "keep_ratio": ratio,
            "cr": round(stats["cr"], 3),
            "file_size_mb": round(comp_mb, 1),
            "file_reduction_pct": round((1 - comp_mb/orig_mb) * 100, 1),
            "params_before_m": round(stats["total_before"]/1e6, 0),
            "params_after_m": round(stats["total_after"]/1e6, 0),
            "loss": round(comp_loss, 4),
            "delta": round(delta, 4),
            "converted": stats["converted"],
        }
        results.append(result)
        
        status = "✅" if abs(delta) < 0.5 else ("⚠️" if abs(delta) < 2.0 else "❌")
        print(f"  {status} ratio={ratio} | CR={stats['cr']:.2f}x | file={orig_mb:.0f}->{comp_mb:.0f}MB ({result['file_reduction_pct']:.0f}%) | delta={delta:+.4f}", flush=True)
        
    except Exception as e:
        print(f"  ❌ ratio={ratio} FAILED: {e}", flush=True)
        results.append({"keep_ratio": ratio, "error": str(e)})
    
    # Free memory
    del model, compressed, loaded
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

# === RESULTS TABLE ===
print(f"\n{'='*80}", flush=True)
print(f"  EXTREME SVD-OBSERVE COMPRESSION SWEEP", flush=True)
print(f"  Model: {MODEL} | {orig_entries/1e6:.0f}M params | {orig_mb:.0f} MB", flush=True)
print(f"{'='*80}", flush=True)
print(f"  {'Ratio':>8} {'CR':>6} {'File':>7} {'Saved':>6} {'Loss':>8} {'Delta':>8}  Status", flush=True)
print(f"  {'-'*8} {'-'*6} {'-'*7} {'-'*6} {'-'*8} {'-'*8}  {'-'*6}", flush=True)
for r in results:
    if "error" in r:
        print(f"  {r['keep_ratio']:>7.0%} {'ERR':>6} {'':>7} {'':>6} {'':>8} {'':>8}  ❌ {r['error']}", flush=True)
    else:
        status = "✅" if abs(r["delta"]) < 0.5 else ("⚠️" if abs(r["delta"]) < 2.0 else "❌")
        print(f"  {r['keep_ratio']:>7.0%} {r['cr']:>5.2f}x {r['file_size_mb']:>5.0f}MB {r['file_reduction_pct']:>+5.1f}% {r['loss']:>8.4f} {r['delta']:>+8.4f}  {status}", flush=True)

# Find breaking point
for i, r in enumerate(results):
    if "error" in r or (abs(r["delta"]) > 1.0 and i > 0):
        prev = results[i-1] if i > 0 else None
        if prev and "delta" in prev:
            print(f"\n  BREAKING POINT: between {prev['keep_ratio']:.0%} (delta={prev['delta']:+.3f}) and {r['keep_ratio']:.0%} (delta={r.get('delta', 'N/A')})", flush=True)
        break

print(f"\n  Time: {time.time()-t0:.0f}s", flush=True)
print(f"{'='*80}", flush=True)

with open("sweep_results.json", "w") as f:
    json.dump({"model": MODEL, "base_loss": base_loss, "results": results}, f, indent=2)
log("Saved sweep_results.json")
