"""weightnoise — Visualize and remove noise from neural network weights.

Usage:
  weightnoise inspect Qwen/Qwen3.5-0.8B          # Show noise report
  weightnoise inspect Qwen/Qwen3.5-0.8B --layer 5 # Single layer detail
  weightnoise prune Qwen/Qwen3.5-0.8B --keep 0.5 # Remove 50% of noise
  weightnoise prune Qwen/Qwen3.5-0.8B --method spectral --threshold 0.1
"""

__version__ = "0.4.0"
