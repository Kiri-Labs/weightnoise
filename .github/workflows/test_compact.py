#!/usr/bin/env python3
"""Compact Qwen3.5-0.8B with SVD factors and verify file size reduction."""
import os, sys, torch
sys.stdout.reconfigure(line_buffering=True)
HF_TOKEN = os.environ.get("HF_TOKEN", "")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from weightnoise.compress import compress_model
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

MODEL = "Qwen/Qwen3.5-0.8B"
OUTPUT = "/tmp/compact_qwen"
t0 = __import__("time").time()
log = lambda m: print(f"[{__import__('time').time()-t0:.0f}s] {m}", flush=True)

log("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
tok = AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN)
base.eval()

log("Compressing...")
compressed, stats = compress_model(base, keep_ratio=0.5, method="svd-observe")

log(f"Stats: {stats}")

# Save the compressed model (truly smaller on disk)
log(f"Saving to {OUTPUT}...")
compressed.save_pretrained(OUTPUT, safe_serialization=False)
AutoTokenizer.from_pretrained(MODEL, token=HF_TOKEN).save_pretrained(OUTPUT)
AutoConfig.from_pretrained(MODEL, token=HF_TOKEN).save_pretrained(OUTPUT)

# Check sizes
import glob
original_dir = "/tmp/orig_qwen"
orig = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, token=HF_TOKEN, low_cpu_mem_usage=True)
orig.save_pretrained(original_dir, safe_serialization=False)
orig_size = sum(os.path.getsize(f) for f in glob.glob(f"{original_dir}/*.bin"))
comp_size = sum(os.path.getsize(f) for f in glob.glob(f"{OUTPUT}/*.bin"))
orig_mb = orig_size / (1024*1024)
comp_mb = comp_size / (1024*1024)

# Count tensor entries to verify actual parameter reduction
orig_entries = sum(p.numel() for p in orig.parameters())
comp_entries = sum(p.numel() for p in compressed.parameters())

print(f"\n{'='*60}", flush=True)
print(f"  SVD COMPACT RESULTS", flush=True)
print(f"{'='*60}", flush=True)
print(f"  Original entries: {orig_entries/1e6:.0f}M tensors", flush=True)
print(f"  Compact entries:  {comp_entries/1e6:.0f}M tensors", flush=True)
print(f"  Original file:    {orig_mb:.1f} MB", flush=True)
print(f"  Compacted file:   {comp_mb:.1f} MB", flush=True)
print(f"  File reduction:   {(1 - comp_mb/orig_mb)*100:.1f}%", flush=True)
print(f"  CR (m params):    {stats['cr']:.2f}x ({stats['total_before']/1e6:.0f}M -> {stats['total_after']/1e6:.0f}M)", flush=True)
print(f"  Converted:        {stats['converted']}", flush=True)
print(f"  Skipped (sq):     {stats['skipped_square']}", flush=True)

if comp_mb < orig_mb * 0.95:
    print(f"  ✅ ACTUAL COMPRESSION: {orig_mb:.1f} -> {comp_mb:.1f} MB ({(1-comp_mb/orig_mb)*100:.1f}% smaller)", flush=True)
elif comp_mb < orig_mb:
    print(f"  ⚠️ MODEST: {orig_mb:.1f} -> {comp_mb:.1f} MB ({(1-comp_mb/orig_mb)*100:.1f}%)", flush=True)
else:
    print(f"  ❌ NO COMPRESSION: {comp_mb:.1f} MB >= {orig_mb:.1f} MB", flush=True)

import shutil
shutil.rmtree(OUTPUT)
shutil.rmtree(original_dir)
print(f"  Time: {__import__('time').time()-t0:.0f}s", flush=True)
