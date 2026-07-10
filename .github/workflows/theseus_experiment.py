#!/usr/bin/env python3
"""Theseus vs SVD: Cross-architecture WIT transfer experiment.
Runs on GHA: streams teacher weights to save memory.
Teacher: Qwen3.5-4B (32L d=2560) -> Student: Qwen3.5-0.8B (24L d=1024)
"""
import os, sys, time, json, torch, gc, re

t0 = time.time()
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["TOKENIZERS_PARALLELISM"] = "false"
log = lambda m: print(f"[{time.time()-t0:.0f}s] {m}") or sys.stdout.flush()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
import safetensors

TEACHER = "Qwen/Qwen3.5-4B"
STUDENT = "Qwen/Qwen3.5-0.8B"

# Configs
t_cfg = AutoConfig.from_pretrained(TEACHER, token=HF_TOKEN).text_config
s_cfg = AutoConfig.from_pretrained(STUDENT, token=HF_TOKEN).text_config
n_t, d_t = t_cfg.num_hidden_layers, t_cfg.hidden_size
n_s, d_s = s_cfg.num_hidden_layers, s_cfg.hidden_size
log(f"Teacher: {n_t}L d={d_t} | Student: {n_s}L d={d_s}")

# Layer mapping (proportional)
ratio = n_t / n_s
mapping = {}
for s_idx in range(n_s):
    t_start = int(round(s_idx * ratio))
    t_end = int(round((s_idx + 1) * ratio))
    mapping[s_idx] = list(range(t_start, max(t_end, t_start + 1), min(n_t)))
log(f"Mapping: {mapping}")

def fast_svd(W, target_shape):
    """SVD projection preserving top spectral components."""
    out_d, in_d = target_shape
    k = min(out_d, in_d, *W.shape)
    U, S, Vh = torch.linalg.svd(W.float(), full_matrices=False)
    return U[:out_d, :k] @ torch.diag(S[:k]) @ Vh[:in_d, :k].T

def theseus_align(t_in, t_out, s_in, s_out):
    """Procrustes alignment from activation cross-covariance."""
    X_T = torch.stack(t_in); X_S = torch.stack(s_in)
    Y_T = torch.stack(t_out); Y_S = torch.stack(s_out)
    X_T -= X_T.mean(0); X_S -= X_S.mean(0)
    Y_T -= Y_T.mean(0); Y_S -= Y_S.mean(0)
    U_in, _, Vh_in = torch.linalg.svd(X_T.T @ X_S, full_matrices=False)
    U_out, _, Vh_out = torch.linalg.svd(Y_T.T @ Y_S, full_matrices=False)
    return U_in @ Vh_in, U_out @ Vh_out

# Download teacher shards
log("Downloading teacher shards...")
shard_dir = snapshot_download(TEACHER, token=HF_TOKEN, allow_patterns="*.safetensors")
shards = sorted(f for f in os.listdir(shard_dir) if f.endswith(".safetensors"))
log(f"{len(shards)} shards")

# Helper: get teacher weight from shards
def get_teacher_weight(name):
    for sf in shards:
        path = os.path.join(shard_dir, sf)
        with safetensors.safe_open(path, framework="pt") as f:
            if name in f.keys():
                return f.get_tensor(name)
    return None

# Test texts
TEXTS = [
    "Future of AI depends on efficient models.",
    "Neural networks learn hierarchical representations.",
    "Weight-space transfer enables cross-architecture knowledge.",
    "Transformers use self-attention to process sequences.",
    "SVD reveals the spectral structure of matrices.",
]

def evaluate(model, tok, texts):
    losses = []
    with torch.no_grad():
        for txt in texts:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=128).input_ids
            losses.append(model(ids, labels=ids).loss.item())
    return sum(losses)/len(losses)

# ============== METHOD 1: SVD ==============
log("=== SVD METHOD ===")
student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(STUDENT, token=HF_TOKEN)
n_st = 0

with torch.no_grad():
    for s_name, s_param in student.named_parameters():
        if s_param.ndim != 2 or "layers." not in s_name: continue
        sm = re.search(r'layers\.(\d+)', s_name)
        if not sm: continue
        s_idx = int(sm.group(1))
        s_sfx = s_name.split(f"layers.{s_idx}.")[-1]
        t_inds = mapping.get(s_idx, [])
        
        # Stream teacher weights for this layer
        tensors = []
        for t_idx in t_inds:
            t_name = f"model.layers.{t_idx}.{s_sfx}"
            tw = get_teacher_weight(t_name)
            if tw is not None: tensors.append(tw.float())
        if not tensors: continue
        
        composed = torch.mean(torch.stack(tensors), dim=0)
        projected = fast_svd(composed, s_param.shape)
        s_param.data.copy_(projected.to(s_param.dtype))
        n_st += 1

log(f"SVD: {n_st} matrices, evaluating...")
svd_loss = evaluate(student, tok, TEXTS)
log(f"SVD loss: {svd_loss:.3f}")
del student; gc.collect()

# ============== METHOD 2: THESEUS ==============
log("=== THESEUS METHOD ===")
student = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)

# Load teacher for activation collection
log("Loading teacher for activations...")
teacher = AutoModelForCausalLM.from_pretrained(TEACHER, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
teacher.eval()
t_params = dict(teacher.named_parameters())

CAL_TEXTS = [
    "Future of AI depends on efficient models.",
    "Neural networks learn hierarchical representations.",
    "Weight-space transfer enables cross-architecture knowledge.",
]

th_n = 0
with torch.no_grad():
    for s_name, s_param in student.named_parameters():
        if s_param.ndim != 2 or "layers." not in s_name: continue
        sm = re.search(r'layers\.(\d+)', s_name)
        if not sm: continue
        s_idx, s_sfx = int(sm.group(1)), s_name.split(f"layers.{s_idx}.")[-1]
        t_inds = mapping.get(s_idx, [])
        tensors = [t_params.get(f"model.layers.{t_idx}.{s_sfx}") for t_idx in t_inds]
        tensors = [t.float() for t in tensors if t is not None]
        if not tensors: continue
        
        # Collect activations (2 cal texts)
        t_in, t_out, s_in, s_out = [], [], [], []
        for txt in CAL_TEXTS[:2]:
            ids = tok(txt, return_tensors="pt", truncation=True, max_length=128)
            hs = teacher(**ids, output_hidden_states=True).hidden_states
            t_in.append(hs[t_inds[0]][0, -1, :].cpu())
            t_out.append(hs[t_inds[0]+1][0, -1, :].cpu())
            hs2 = student(**ids, output_hidden_states=True).hidden_states
            s_in.append(hs2[s_idx][0, -1, :].cpu())
            s_out.append(hs2[s_idx+1][0, -1, :].cpu())
        
        if len(t_in) >= 2:
            Ti, To = theseus_align(t_in, t_out, s_in, s_out)
            composed = torch.mean(torch.stack(tensors), dim=0)
            projected = To.T @ composed @ Ti
            projected = fast_svd(projected, s_param.shape)
        else:
            composed = torch.mean(torch.stack(tensors), dim=0)
            projected = fast_svd(composed, s_param.shape)
        
        s_param.data.copy_(projected.to(s_param.dtype))
        th_n += 1

log(f"Theseus: {th_n} matrices, evaluating...")
th_loss = evaluate(student, tok, TEXTS)
log(f"Theseus loss: {th_loss:.3f}")
del student, teacher; gc.collect()

# ============== BASELINE ==============
log("=== BASELINE ===")
orig = AutoModelForCausalLM.from_pretrained(STUDENT, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
o_loss = evaluate(orig, tok, TEXTS)
log(f"Original loss: {o_loss:.3f}")

# ============== RESULTS ==============
print("\n" + "="*60)
print(f"  TEACHER: {TEACHER} ({n_t}L d={d_t})")
print(f"  STUDENT: {STUDENT} ({n_s}L d={d_s})")
print("="*60)
print(f"  Method        | Avg Loss | Gap to Orig")
print(f"  --------------|----------|------------")
print(f"  SVD (baseline)| {svd_loss:.3f}   | +{svd_loss - o_loss:.3f}")
print(f"  Theseus       | {th_loss:.3f}   | +{th_loss - o_loss:.3f}")
print(f"  Original      | {o_loss:.3f}   | 0")
print(f"  THESEUS vs SVD: {svd_loss - th_loss:+.3f} {'✅ better' if th_loss < svd_loss else '❌ worse'}")
print(f"  Total time: {time.time()-t0:.0f}s")
print("="*60)

# Save results
r = {"teacher":TEACHER,"student":STUDENT,"time_s":time.time()-t0,
     "svd_loss":svd_loss,"theseus_loss":th_loss,"original_loss":o_loss,
     "theseus_vs_svd":svd_loss-th_loss,"theseus_gap":th_loss-o_loss}
with open("results.json","w") as f: json.dump(r,f,indent=2)
log("Done! Results in results.json")
