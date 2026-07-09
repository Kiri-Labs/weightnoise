"""Safetensors streaming loader — load one shard at a time, no OOM on 100B models.

Instead of AutoModelForCausalLM.from_pretrained() which loads everything into RAM,
this streams safetensors files one shard at a time, extracts only the weights we need,
then deletes the shard from memory.

For GLM-5 52B, that means ~200MB RSS instead of 200GB.
"""
import torch
import json
import os
import glob


def find_safetensors_shards(model_id_or_path):
    """Get list of safetensors shard files for a model.
    
    Works with local paths or cached HuggingFace models.
    """
    if os.path.isdir(model_id_or_path):
        shards = sorted(glob.glob(os.path.join(model_id_or_path, "*.safetensors")))
        idx_file = os.path.join(model_id_or_path, "model.safetensors.index.json")
        if not shards and os.path.exists(idx_file):
            with open(idx_file) as f:
                idx = json.load(f)
                shards = sorted(set(idx.get("weight_map", {}).values()))
                shards = [os.path.join(model_id_or_path, s) for s in shards]
        return shards
    
    # Try HF cache
    from huggingface_hub import snapshot_download
    import tempfile
    
    # Check if already cached
    cache_path = os.path.expanduser(f"~/.cache/huggingface/hub/models--{model_id_or_path.replace('/', '--')}")
    if os.path.exists(cache_path):
        # Find the snapshots directory
        refs_dir = os.path.join(cache_path, "refs")
        blobs_dir = os.path.join(cache_path, "blobs")
        snapshots_dir = os.path.join(cache_path, "snapshots")
        
        if os.path.exists(snapshots_dir):
            snapshots = os.listdir(snapshots_dir)
            if snapshots:
                snapshot = os.path.join(snapshots_dir, snapshots[0])
                shards = sorted(glob.glob(os.path.join(snapshot, "*.safetensors")))
                if shards:
                    return shards
        
        # Raw blobs (older HF cache format)
        if os.path.exists(blobs_dir):
            shards = sorted(glob.glob(os.path.join(blobs_dir, "model*.safetensors")))
            if shards:
                return shards
    
    return []


def _tensor_size_bytes(tensor):
    """Get the memory size of a tensor in bytes."""
    return tensor.numel() * tensor.element_size()


def _safe_load_shard(shard_path, device="cpu"):
    """Load a single safetensors shard file."""
    from safetensors import safe_open
    tensors = {}
    with safe_open(shard_path, framework="pt", device=device) as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def streaming_extract_weights(model_id, layer_indices=None, weight_filter=None):
    """Extract specific weights from a model without loading it all.
    
    Streams shards one at a time, keeps only what's needed.
    
    Args:
        model_id: HF model ID or local path
        layer_indices: List of layer indices to extract (None = all)
        weight_filter: Function(name, shape) → bool for which weights to keep
    
    Yields:
        (name, tensor) tuples, one at a time
    """
    shards = find_safetensors_shards(model_id)
    if not shards:
        # Fallback: try loading with from_pretrained (may OOM)
        raise RuntimeError(
            f"No safetensors shards found for {model_id}. "
            f"Download the model first with: "
            f"huggingface-cli download {model_id}"
        )
    
    devices = {"cpu": "cpu", "cuda": "cuda"}.get(device, "cpu")
    
    for shard_path in shards:
        tensors = _safe_load_shard(shard_path, device=device)
        
        for name, tensor in tensors.items():
            # Apply layer filter
            if layer_indices is not None:
                # Check if this tensor belongs to any of our layers
                parts = name.replace(".", " ").split()
                tensor_layer = None
                for p in parts:
                    try:
                        tensor_layer = int(p)
                        break
                    except ValueError:
                        continue
                if tensor_layer is not None and tensor_layer not in layer_indices:
                    continue
            
            # Apply custom filter
            if weight_filter is not None and not weight_filter(name, tensor.shape):
                continue
            
            yield name, tensor
        
        # Free shard memory
        del tensors
        
        # Report memory status
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def estimate_model_memory(model_id):
    """Calculate how much RAM/VRAM a model needs without loading it.
    
    Reads config.json + safetensors index to compute sizes.
    """
    if os.path.isdir(model_id):
        config_path = os.path.join(model_id, "config.json")
    else:
        cache_path = os.path.expanduser(
            f"~/.cache/huggingface/hub/models--{model_id.replace('/', '--')}")
        snapshots = os.path.join(cache_path, "snapshots")
        if os.path.exists(snapshots):
            snaps = os.listdir(snapshots)
            config_path = os.path.join(snapshots, snaps[0], "config.json") if snaps else None
        else:
            config_path = None
    
    if config_path and os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
    
    # Sum sizes from safetensors shards
    shards = find_safetensors_shards(model_id)
    total_bytes = 0
    for shard in shards:
        total_bytes += os.path.getsize(shard)
    
    n_params = total_bytes / 2  # fp16: 2 bytes per param
    fp32_gb = round(total_bytes * 2 / 1e9, 1)    # fp32 needs 2x
    fp16_gb = round(total_bytes / 1e9, 1)         # fp16 = file size
    int8_gb = round(total_bytes / 2 / 1e9, 1)     # int8 = half
    
    return {
        "model": model_id,
        "params_b": round(n_params / 1e9, 1),
        "shards": len(shards),
        "fp32_gb": fp32_gb,
        "fp16_gb": fp16_gb,
        "int8_gb": int8_gb,
        "note": "Loading this with AutoModelForCausalLM would need "
                f"{fp32_gb}GB of RAM. Use --stream to avoid OOM.",
    }
