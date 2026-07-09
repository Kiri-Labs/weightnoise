"""WIT Compress: Qwen3.6-27B -> Qwen3.5-0.8B via spectral stitching.
Triggered by GitHub Actions workflow wit-compress.yml"""
import os, json, torch, gc, time, sys

HF_TOKEN = os.environ.get("HF_TOKEN", "")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not set")

from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from safetensors import safe_open

t0 = time.time()
def log(m):
    elapsed = time.time() - t0
    print(f"[{elapsed:.0f}s] {m}")
    sys.stdout.flush()

log("Loading student Qwen3.5-0.8B...")
student = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3.5-0.8B", torch_dtype=torch.float16,
    token=HF_TOKEN, low_cpu_mem_usage=True)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-0.8B", token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

cfg_t = AutoConfig.from_pretrained("Qwen/Qwen3.6-27B", token=HF_TOKEN).text_config
cfg_s = AutoConfig.from_pretrained("Qwen/Qwen3.5-0.8B", token=HF_TOKEN).text_config
d_t, n_t = cfg_t.hidden_size, cfg_t.num_hidden_layers
d_s, n_s = cfg_s.hidden_size, cfg_s.num_hidden_layers
ratio = n_t / n_s
log(f"Teacher: {n_t}L d={d_t} | Student: {n_s}L d={d_s} | Ratio: {ratio:.1f}x")

# Layer mapping
mapping = {}
for s_idx in range(n_s):
    t_start = int(round(s_idx * ratio))
    t_end = int(round((s_idx + 1) * ratio))
    t_end = max(t_end, t_start + 1)
    mapping[s_idx] = list(range(t_start, min(t_end, n_t)))

log(f"Layer mapping: L0<-{mapping[0]}, L{n_s-1}<-{mapping[n_s-1]}")

# Teacher shards
api = HfApi(token=HF_TOKEN)
files = api.list_repo_files("Qwen/Qwen3.6-27B")
shards = sorted([f for f in files if f.endswith(".safetensors")])
log(f"Teacher shards: {len(shards)}")

# Build layer->shard map
layer_shard = {}
for idx, shard_name in enumerate(shards):
    sp = hf_hub_download("Qwen/Qwen3.6-27B", shard_name, token=HF_TOKEN)
    with safe_open(sp, framework="pt", device="cpu") as f:
        for key in f.keys():
            for p in key.split("."):
                if p.isdigit() and int(p) < n_t:
                    lidx = int(p)
                    if lidx not in layer_shard:
                        layer_shard[lidx] = shard_name
    os.remove(sp)
    gc.collect()
    if (idx + 1) % 10 == 0:
        log(f"  Indexed {idx+1}/{len(shards)} shards ({len(layer_shard)} layers)")

log(f"Indexed {len(layer_shard)}/{n_t} layers")

# Compose
student_params = dict(student.named_parameters())
n_stitched, n_skipped = 0, 0

for s_idx in range(n_s):
    t_indices = mapping[s_idx]
    teacher_mats = {}
    
    for t_idx in t_indices:
        shard_name = layer_shard.get(t_idx)
        if not shard_name: continue
        sp = hf_hub_download("Qwen/Qwen3.6-27B", shard_name, token=HF_TOKEN)
        with safe_open(sp, framework="pt", device="cpu") as f:
            for key in f.keys():
                if f"layers.{t_idx}." in key and key.endswith(".weight"):
                    t = f.get_tensor(key)
                    if t.ndim == 2:
                        teacher_mats.setdefault(key, []).append(t.clone())
        os.remove(sp)
        gc.collect()
    
    for s_name, s_param in student_params.items():
        if s_param.ndim != 2: continue
        if f"layers.{s_idx}." not in s_name: continue
        
        suffix = s_name.split(f"layers.{s_idx}.")[-1]
        matches = [k for k in teacher_mats if k.endswith(suffix)]
        if not matches:
            n_skipped += 1
            continue
        weights = teacher_mats[matches[0]]
        if not weights:
            n_skipped += 1
            continue
        
        # Mean-average teacher weights
        composed = torch.mean(torch.stack([w.float() for w in weights]), dim=0)
        
        # SVD-project to student dimensions
        out_dim, in_dim = s_param.shape
        k = min(out_dim, in_dim)
        U, S, V = torch.svd_lowrank(composed, q=k, niter=2)
        projected = U[:out_dim, :k] @ torch.diag(S[:k]) @ V[:in_dim, :k].T
        
        with torch.no_grad():
            s_param.data.copy_(projected.to(s_param.dtype))
        n_stitched += 1
    
    if (s_idx + 1) % 4 == 0 or s_idx == n_s - 1:
        log(f"Layer {s_idx+1}/{n_s}: {n_stitched} stitched")

log(f"Composition complete: {n_stitched} matrices, {n_skipped} skipped")

# Save
os.makedirs("/tmp/wit-model", exist_ok=True)
student.save_pretrained("/tmp/wit-model")
tokenizer.save_pretrained("/tmp/wit-model")
log("Model saved locally")

# Upload to HF
log("Uploading to HuggingFace Hub...")
api.create_repo("KiriLabs/Qwen3.6-0.8B-WIT-27B-Transfer", exist_ok=True, private=False)
api.upload_folder(folder_path="/tmp/wit-model", repo_id="KiriLabs/Qwen3.6-0.8B-WIT-27B-Transfer")
log("Model weights uploaded")

# Model card
log("Uploading model card...")
card = f"""---
tags:
- wit
- weight-space-intelligence-transfer
- cross-architecture
- qwen
metrics:
- compression
---

# Qwen3.6-WIT-27B-Transfer

**Weight-Space Intelligence Transfer**: Qwen3.6-27B -> Qwen3.5-0.8B

## Architecture

| Property | Value |
|----------|-------|
| Teacher | Qwen3.6-27B ({n_t}L, d={d_t}) |
| Student | Qwen3.5-0.8B ({n_s}L, d={d_s}) |
| Layer ratio | {ratio:.1f}x |
| Matrices composed | {n_stitched} of {n_stitched + n_skipped} |

## Method

For each student layer, teacher layers are mean-averaged, then SVD-projected to
student dimensions. This physically changes matrix sizes.

- d_model: {d_t} -> {d_s} (truncate to {d_s/d_t:.0%} of spectrum)
- intermediate: 17408 -> 3584
- layers: {n_t} -> {n_s}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "KiriLabs/Qwen3.6-0.8B-WIT-27B-Transfer"
)
tokenizer = AutoTokenizer.from_pretrained(
    "KiriLabs/Qwen3.6-0.8B-WIT-27B-Transfer"
)
```

*Research checkpoint — fine-tuning needed for production quality.*

## Related

- [weightnoise](https://github.com/Kiri-Labs/weightnoise) (CLI compression tool)
- [WIT-CrossSize-Transport](https://huggingface.co/KiriLabs/WIT-CrossSize-Transport)
"""
api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                repo_id="KiriLabs/Qwen3.6-0.8B-WIT-27B-Transfer")
log(f"DONE: https://huggingface.co/KiriLabs/Qwen3.6-0.8B-WIT-27B-Transfer")
