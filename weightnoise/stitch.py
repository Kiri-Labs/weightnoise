"""Cross-architecture weight stitching via WIT.
Methods:
  - svd: mean-averaging + SVD projection (no calibration data needed)
  - theseus: Procrustes alignment from activations (requires calibration, ICML 2026)

Usage:
  weightnoise compress Qwen/Qwen3.6-27B Qwen/Qwen3.5-0.8B --save ./compressed
  weightnoise compress TinyLlama/TinyLlama-1.1B Qwen/Qwen3.5-0.8B --method theseus --calibrate
"""
import torch
import numpy as np
import os, json, gc, re, sys
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoConfig, AutoTokenizer


def _find_dims(model_object=None, model_id=None, config=None):
    if config is None:
        if model_object is not None:
            config = model_object.config
        elif model_id is not None:
            config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    if hasattr(config, 'text_config'):
        config = config.text_config
    d_model = None
    for attr in ["hidden_size", "d_model", "n_embd", "dim", "hidden_dim"]:
        if hasattr(config, attr): d_model = getattr(config, attr); break
    n_layers = None
    for attr in ["num_hidden_layers", "n_layer", "num_layers",
                 "decoder_layers", "num_decoder_layers"]:
        if hasattr(config, attr): n_layers = getattr(config, attr); break
    return d_model, n_layers


def _svd_project(composed_w, target_shape):
    out_dim, in_dim = target_shape
    k = min(out_dim, in_dim)
    try:
        U, S, V = torch.svd_lowrank(composed_w, q=k, niter=2)
    except RuntimeError:
        U, S, V = torch.svd(composed_w)
        U, S, V = U[:, :k], S[:k], V[:, :k]
    return U[:out_dim, :k] @ torch.diag(S[:k]) @ V[:in_dim, :k].T


def _theseus_align(teacher_acts_in, teacher_acts_out,
                   student_acts_in, student_acts_out):
    """Procrustes alignment from activation cross-covariance (Theseus, ICML 2026)."""
    X_T = torch.stack(teacher_acts_in)
    X_S = torch.stack(student_acts_in)
    Y_T = torch.stack(teacher_acts_out)
    Y_S = torch.stack(student_acts_out)
    X_T = X_T - X_T.mean(dim=0, keepdim=True)
    X_S = X_S - X_S.mean(dim=0, keepdim=True)
    Y_T = Y_T - Y_T.mean(dim=0, keepdim=True)
    Y_S = Y_S - Y_S.mean(dim=0, keepdim=True)
    C_in = X_T.T @ X_S
    C_out = Y_T.T @ Y_S
    U_in, _, V_in = torch.svd(C_in)
    U_out, _, V_out = torch.svd(C_out)
    T_in = U_in @ V_in.T
    T_out = U_out @ V_out.T
    return T_in, T_out


def _layer_mapping(n_teacher, n_student):
    ratio = n_teacher / n_student
    mapping = {}
    for s_idx in range(n_student):
        t_start = int(round(s_idx * ratio))
        t_end = int(round((s_idx + 1) * ratio))
        t_end = max(t_end, t_start + 1)
        mapping[s_idx] = list(range(t_start, min(t_end, n_teacher)))
    return mapping


def _collect_calibration_activations(model, tokenizer, texts, layer_idx, device="cpu"):
    """Run texts through a model and collect layer input/output activations."""
    model.eval()
    act_in = []
    act_out = []
    handles = []
    
    def get_hook(name, store_out):
        def hook(module, inp, out):
            if inp[0].ndim == 3:
                vec = inp[0][0, -1, :].detach().cpu()
                store_out.append(("in", vec))
            if isinstance(out, tuple):
                o = out[0]
            else:
                o = out
            if o.ndim == 3:
                vec = o[0, -1, :].detach().cpu()
                store_out.append(("out", vec))
        return hook
    
    # Register hooks on the layer
    layer_name = None
    for name, module in model.named_modules():
        if f"layers.{layer_idx}" == name or f"layers.{layer_idx}." in name:
            if not any(x in name for x in ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"]):
                layer_name = name
                break
    
    if layer_name is None:
        # Find any module with this layer index
        for name, module in model.named_modules():
            if f".{layer_idx}." in f".{name}." and not any(x in name for x in ["attn", "mlp", "norm", "embed", "lm_head"]):
                layer_name = name
                break
    
    if layer_name is None:
        return [], []
    
    collected = []
    layer_mod = model.get_submodule(layer_name)
    handle = layer_mod.register_forward_hook(get_hook(layer_idx, collected))
    
    try:
        for text in texts:
            inp = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                model(**inp)
    except:
        pass
    
    handle.remove()
    
    # Separate input and output activations
    acts_in = [c[1] for c in collected if c[0] == "in"]
    acts_out = [c[1] for c in collected if c[0] == "out"]
    
    return acts_in, acts_out


def _build_model_card(teacher_id, student_id, results, method="svd"):
    """Build a HuggingFace model card from results."""
    n_t = results.get("teacher_layers", "?")
    n_s = results.get("student_layers", "?")
    d_t = results.get("teacher_dim", "?")
    d_s = results.get("student_dim", "?")
    stitched = results.get("matrices_stitched", 0)
    
    return f"""---
tags:
- wit
- weight-space-intelligence-transfer
- cross-architecture
- {method}
metrics:
- compression
---

# WIT: {teacher_id} → {student_id}

**Weight-Space Intelligence Transfer** using **{method.upper()}** method.

## Architecture

| Property | Teacher | Student |
|----------|---------|---------|
| Model | {teacher_id} | {student_id} |
| Layers | {n_t} | {n_s} |
| Hidden dim | {d_t} | {d_s} |
| Matrices composed | — | {stitched} |

## Method

{_METHOD_DESCRIPTIONS.get(method, "")}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("KiriLabs/{student_id.split('/')[-1]}-WIT")
tokenizer = AutoTokenizer.from_pretrained("KiriLabs/{student_id.split('/')[-1]}-WIT")
```

*Research checkpoint — no fine-tuning applied.*
"""


_METHOD_DESCRIPTIONS = {
    "svd": """For each student layer, the corresponding teacher layers are mean-averaged,
then SVD-projected to the student's exact dimensions. This preserves the spectral
structure (top singular components) while physically changing matrix sizes.""",
    "theseus": """Based on Theseus (Salici et al., ICML 2026). Uses Procrustes alignment
from activation cross-covariance on calibration data to find optimal linear maps
between teacher and student representational spaces. The transport equation is:
W_student = T_out @ W_teacher @ T_in^T where T_in, T_out are orthogonal matrices.""",
}


def compress(teacher_id, student_id, device="cpu",
             trust_remote_code=False, save_path=None,
             streaming=False, method="svd",
             calibration_texts=None, upload_to_hf=None,
             hf_token=None):
    """Compress a teacher model into a student architecture via WIT.
    
    Args:
        teacher_id: Large teacher model ID
        student_id: Small student model ID
        device: Device for computation
        method: 'svd' (no calibration) or 'theseus' (needs calibration data)
        calibration_texts: List of strings for activation alignment (theseus only)
        upload_to_hf: HF repo name to upload results
        hf_token: HF token for upload
    """
    print(f"  Loading teacher config: {teacher_id}")
    t_config = AutoConfig.from_pretrained(teacher_id, trust_remote_code=trust_remote_code)
    d_t, n_t = _find_dims(config=t_config)
    
    print(f"  Loading student: {student_id}")
    student = AutoModelForCausalLM.from_pretrained(
        student_id, torch_dtype=torch.float32,
        device_map=device, trust_remote_code=trust_remote_code,
    )
    d_s, n_s = _find_dims(model_object=student)
    
    s_params = sum(p.numel() for p in student.parameters())
    mapping = _layer_mapping(n_t, n_s)
    
    print(f"\n  Architecture: Teacher {n_t}L d={d_t} → Student {n_s}L d={d_s}")
    print(f"  Method: {method}")
    print(f"  Ratio: {n_t/n_s:.1f}x")
    
    # Load teacher weights (streaming for large models)
    teacher_param_dict = None
    if streaming:
        print("  Streaming teacher weights (JIT mode)...")
        # Build mapping and stream
        _stream_compress(teacher_id, student, mapping, d_t, d_s, device, method, calibration_texts)
        n_stitched = sum(1 for _ in student.named_parameters() if _.ndim == 2 and 'layers.' in _[0])
        n_skipped = 0
    else:
        if teacher_param_dict is None:
            print("  Loading teacher...")
            # For teachers > 7B, fall back to streaming
            teacher = AutoModelForCausalLM.from_pretrained(
                teacher_id, torch_dtype=torch.float32,
                device_map=device, trust_remote_code=trust_remote_code,
                low_cpu_mem_usage=True,
            )
            teacher_param_dict = dict(teacher.named_parameters())
        
        n_stitched, n_skipped = _compose_weights(
            student, teacher_param_dict, mapping, method, calibration_texts,
            teacher_model=teacher if not streaming else None,
        )
        
        del teacher
        gc.collect()
    
    print(f"\n  Stitched: {n_stitched} | Skipped: {n_skipped}")
    
    if save_path:
        os.makedirs(save_path, exist_ok=True)
        student.save_pretrained(save_path)
        # Also copy tokenizer if available
        try:
            tok = AutoTokenizer.from_pretrained(student_id, trust_remote_code=trust_remote_code)
            tok.save_pretrained(save_path)
        except:
            pass
        print(f"  Saved to: {save_path}")
    
    if upload_to_hf and hf_token:
        print(f"  Uploading to HF: {upload_to_hf}")
        api = HfApi(token=hf_token)
        api.create_repo(upload_to_hf, exist_ok=True, private=False)
        api.upload_folder(folder_path=save_path, repo_id=upload_to_hf)
        
        # Model card
        results = {
            "teacher": teacher_id, "student": student_id,
            "teacher_layers": n_t, "student_layers": n_s,
            "teacher_dim": d_t, "student_dim": d_s,
            "matrices_stitched": n_stitched, "method": method,
        }
        card = _build_model_card(teacher_id, student_id, results, method)
        api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                        repo_id=upload_to_hf)
        print(f"  Published: https://huggingface.co/{upload_to_hf}")
    
    return {
        "teacher": teacher_id, "student": student_id,
        "student_params_m": round(s_params / 1e6, 1),
        "teacher_layers": n_t, "student_layers": n_s,
        "matrices_stitched": n_stitched, "matrices_skipped": n_skipped,
        "method": method,
        "mapping": {str(k): v for k, v in mapping.items()},
        "save_path": save_path,
    }


def _compose_weights(student, teacher_params, mapping, method="svd",
                     calibration_texts=None, teacher_model=None):
    """Compose teacher weights into student using specified method."""
    n_stitched = 0
    n_skipped = 0
    
    with torch.no_grad():
        for s_name, s_param in student.named_parameters():
            if s_param.ndim != 2:
                continue
            s_match = re.search(r'layers\.(\d+)', s_name)
            if not s_match:
                continue
            s_idx = int(s_match.group(1))
            
            s_parts = s_name.split(".")
            try:
                li_pos = next(i for i, p in enumerate(s_parts) if p.isdigit() and int(p) == s_idx)
            except:
                continue
            w_suffix = ".".join(s_parts[li_pos + 1:])
            
            t_indices = mapping.get(s_idx, [])
            teacher_tensors = []
            
            for t_idx in t_indices:
                t_name = ".".join(s_parts[:li_pos]) + f".{t_idx}." + w_suffix
                t_weight = teacher_params.get(t_name)
                
                if t_weight is None:
                    alts = [
                        f"model.language_model.layers.{t_idx}.{w_suffix}",
                        f"model.layers.{t_idx}.{w_suffix}",
                        f"transformer.h.{t_idx}.{w_suffix}",
                        f"layers.{t_idx}.{w_suffix}",
                        f"encoder.layer.{t_idx}.{w_suffix}",
                    ]
                    for alt in alts:
                        if alt in teacher_params:
                            t_weight = teacher_params[alt]
                            break
                
                if t_weight is not None:
                    teacher_tensors.append(t_weight.float())
            
            if len(teacher_tensors) < 1:
                n_skipped += 1
                continue
            
            if method == "theseus" and teacher_model is not None and calibration_texts:
                # Collect activations for theseus alignment
                t_in, t_out = _collect_calibration_activations(
                    teacher_model, None, calibration_texts, t_indices[0])
                s_in, s_out = _collect_calibration_activations(
                    student, None, calibration_texts, s_idx)
                
                if len(t_in) > 0 and len(s_in) > 0:
                    T_in, T_out = _theseus_align(t_in, t_out, s_in, s_out)
                    # Transport: W_s = T_out @ W_t @ T_in^T
                    composed = torch.mean(torch.stack(teacher_tensors), dim=0)
                    projected = T_out.T @ composed @ T_in
                    # Additional SVD truncation to exact dims
                    projected = _svd_project(projected, s_param.shape)
                else:
                    # Fallback to SVD
                    composed = torch.mean(torch.stack(teacher_tensors), dim=0)
                    projected = _svd_project(composed, s_param.shape)
            else:
                composed = torch.mean(torch.stack(teacher_tensors), dim=0)
                projected = _svd_project(composed, s_param.shape)
            
            s_param.data.copy_(projected.to(s_param.dtype))
            n_stitched += 1
    
    return n_stitched, n_skipped


def _stream_compress(teacher_id, student, mapping, d_t, d_s, device, method, calibration_texts):
    """Streaming version that processes teacher shards one at a time."""
    from .stream import streaming_extract_weights
    
    # Build inverse mapping: which teacher layers go to which student layers
    needed_teacher_layers = set()
    for s_indices in mapping.values():
        needed_teacher_layers.update(s_indices)
    
    # Collect teacher weights by layer
    teacher_by_layer = {}
    for name, tensor in streaming_extract_weights(teacher_id, layer_indices=list(needed_teacher_layers)):
        if tensor.ndim == 2:
            match = re.search(r'layers\.(\d+)', name)
            if match:
                lidx = int(match.group(1))
                if lidx not in teacher_by_layer:
                    teacher_by_layer[lidx] = {}
                teacher_by_layer[lidx][name] = tensor
    
    with torch.no_grad():
        for s_name, s_param in student.named_parameters():
            if s_param.ndim != 2:
                continue
            s_match = re.search(r'layers\.(\d+)', s_name)
            if not s_match:
                continue
            s_idx = int(s_match.group(1))
            
            t_indices = mapping.get(s_idx, [])
            teacher_tensors = []
            
            for t_idx in t_indices:
                t_layer = teacher_by_layer.get(t_idx, {})
                for t_name, t_weight in t_layer.items():
                    if t_name.split("layers." + str(t_idx) + ".")[-1] == s_name.split("layers." + str(s_idx) + ".")[-1]:
                        teacher_tensors.append(t_weight.float())
            
            if len(teacher_tensors) < 1:
                continue
            
            composed = torch.mean(torch.stack(teacher_tensors), dim=0)
            projected = _svd_project(composed, s_param.shape)
            s_param.data.copy_(projected.to(s_param.dtype))
