#!/usr/bin/env python3
"""Extreme SVD compression sweep: test keep_ratios 50% down to 1%."""
import os, sys, json, time

HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sys.stdout.reconfigure(line_buffering=True)
t0 = time.time()
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}", flush=True)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load once, compress at different ratios, evaluate
MODEL = "Qwen/Qwen3.5-0.8B"
log("Loading model...")
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN)
tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
base.eval()

# Reference size
orig_entries = sum(p.numel() for p in base.parameters())
short_texts = [
    "The future of AI depends on efficient models that run on local hardware.",
    "Neural networks learn hierarchical representations from data.",
    "Weight-space intelligence transfer enables cross-architecture knowledge.",
    "Quantum computing may revolutionize machine learning in the coming decades.",
    "Large language models require significant computational resources."
]

def avg_loss(model, tok, texts, max_len=128):
    losses = []
    model.eval()
    with torch.no_grad():
        for txt in texts:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=max_len).input_ids
            losses.append(model(ids, labels=ids).loss.item())
    return sum(losses)/len(losses) if losses else 0.0

base_loss = avg_loss(base, tok, short_texts)
log(f"Base loss: {base_loss:.4f} ({orig_entries/1e6:.0f}M entries)")

# SVD factorize function (self-contained)
def factorize_model(model, keep_ratio, method="svd-observe"):
    """Convert Linear layers to SVDLinear."""
    import torch.nn as nn
    
    class SVDLinear(nn.Module):
        def __init__(self, in_f, out_f, k, bias=False):
            super().__init__()
            self.U = nn.Parameter(torch.empty(out_f, k))
            self.S = nn.Parameter(torch.empty(k))
            self.Vt = nn.Parameter(torch.empty(k, in_f))
            self.bias = nn.Parameter(torch.empty(out_f)) if bias else None
        
        def forward(self, x):
            # Handle arbitrary input dims (batch, ..., in_f) -> (batch, ..., out_f)
            shape = x.shape[:-1]
            x_flat = x.reshape(-1, x.shape[-1])
            out = torch.mm(torch.mm(x_flat, self.Vt.T) * self.S.unsqueeze(0), self.U.T)
            if self.bias is not None:
                out = out + self.bias
            return out.reshape(*shape, -1)
    
    conv_stats = {"before": 0, "after": 0, "converted": 0, "skipped": 0}
    
    def convert(module, name=""):
        for c_name, child in list(module.named_children()):
            if isinstance(child, nn.Linear) and child.weight.ndim == 2:
                m, n = child.weight.shape
                conv_stats["before"] += m * n
                k_keep = max(1, int(min(m, n) * keep_ratio))
                stored = k_keep * (m + n)
                
                if stored < m * n:
                    W = child.weight.data.float()
                    U, S, Vt = torch.linalg.svd(W, full_matrices=False)
                    k_total = len(S)
                    if method == "svd-observe":
                        sal = S ** 4
                        _, idx = sal.sort(descending=True)
                        keep = torch.zeros(k_total, dtype=torch.bool)
                        keep[idx[:k_keep]] = True
                        kept = keep.nonzero(as_tuple=True)[0]
                        removed = (~keep).nonzero(as_tuple=True)[0]
                        S_new = S.clone()
                        if len(removed) > 0:
                            Un = U / (U.norm(dim=0, keepdim=True) + 1e-8)
                            Vn = Vt.T / (Vt.T.norm(dim=0, keepdim=True) + 1e-8)
                            for p in removed:
                                c = (Un[:, kept].T @ Un[:, p:p+1]).squeeze() * (Vn[:, kept].T @ Vn[:, p:p+1]).squeeze()
                                S_new[kept] += c * S[p] / (1 + c.abs().mean() + 1e-8)
                        U, S, Vt = U[:, kept], S_new[kept], Vt[kept, :]
                        k_keep = len(kept)

                    svd = SVDLinear(n, m, k_keep, bias=child.bias is not None)
                    svd.U.data = U[:, :k_keep]
                    svd.S.data = S[:k_keep]
                    svd.Vt.data = Vt[:k_keep, :]
                    if child.bias is not None:
                        svd.bias.data = child.bias.data
                    setattr(module, c_name, svd)
                    conv_stats["converted"] += 1
                    conv_stats["after"] += k_keep * (m + n)
                else:
                    conv_stats["after"] += m * n
                    conv_stats["skipped"] += 1
            elif not isinstance(child, (nn.Linear, SVDLinear)):
                convert(child, f"{name}.{c_name}" if name else c_name)
    
    convert(model)
    cr = conv_stats["before"] / max(conv_stats["after"], 1)
    return model, {**conv_stats, "cr": round(cr, 3)}


ratios = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
results = []

for ratio in ratios:
    log(f"\n=== keep_ratio={ratio} ===")
    try:
        # Load fresh model
        model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN)
        compressed, stats = factorize_model(model, keep_ratio=ratio)
        comp_loss = avg_loss(compressed, tok, short_texts)
        delta = comp_loss - base_loss
        
        comp_entries = sum(p.numel() for p in compressed.parameters())
        status = "✅" if abs(delta) < 0.5 else ("⚠️" if abs(delta) < 2.0 else "❌")
        
        log(f"  {status} CR={stats['cr']}x entries={comp_entries/1e6:.0f}M delta={delta:+.4f}")
        
        results.append({
            "keep_ratio": ratio,
            "cr": stats["cr"],
            "entries_before_m": round(orig_entries/1e6, 1),
            "entries_after_m": round(comp_entries/1e6, 1),
            "file_estimate": f"~{comp_entries*2/(1024*1024*1024):.2f}GB",
            "loss": round(comp_loss, 4),
            "delta": round(delta, 4),
            "status": status,
            "converted": stats["converted"],
            "skipped": stats["skipped"]
        })
        
        del model, compressed
        if delta > 2.0 and ratio < 0.2:
            log(f"  Breaking at {ratio}, stopping")
            if ratio < 0.5:
                break
    except Exception as e:
        import traceback
        log(f"  ❌ FAILED: {e}")
        log(traceback.format_exc())
        results.append({"keep_ratio": ratio, "error": str(e)})

# Print results table
print(f"\n{'='*70}", flush=True)
print(f"  EXTREME SVD-OBSERVE COMPRESSION: {MODEL}", flush=True)
print(f"{'='*70}", flush=True)
print(f"  {'Ratio':>7} {'CR':>6} {'Entries':>9} {'Size':>6} {'Loss':>8} {'Delta':>9}  Status", flush=True)
print(f"  {'-'*7} {'-'*6} {'-'*9} {'-'*6} {'-'*8} {'-'*9}  {'-'*6}", flush=True)
for r in results:
    if "error" in r:
        print(f"  {r['keep_ratio']:>6.0%} {'ERR':>6} {'':>9} {'':>6} {'':>8} {'':>9}  ❌ {r['error']}", flush=True)
    else:
        print(f"  {r['keep_ratio']:>6.0%} {r['cr']:>5.2f}x {r['entries_after_m']:>5.0f}M {r['file_estimate']:>7s} {r['loss']:>8.4f} {r['delta']:>+8.4f}  {r['status']}", flush=True)

# Find breakpoint
for i, r in enumerate(results):
    if "error" in r or (abs(r.get("delta", 0)) > 1.0 and i > 0):
        prev = results[i-1] if i > 0 else None
        if prev and "delta" in prev:
            print(f"\n  BREAKPOINT: between {prev['keep_ratio']:.0%} (delta={prev['delta']:+.3f}) and {r['keep_ratio']:.0%} (delta={r.get('delta', 'N/A')})", flush=True)
        break
else:
    print(f"\n  No breakpoint found — all ratios within tolerance", flush=True)

print(f"  Time: {time.time()-t0:.0f}s", flush=True)
print(f"{'='*70}", flush=True)

with open("sweep_results.json", "w") as f:
    json.dump({"model": MODEL, "base_loss": base_loss, "results": results}, f, indent=2)
log("Saved sweep_results.json")
