"""weightnoise — Visualize and remove noise from neural network weights.

Usage:
  weightnoise inspect Qwen/Qwen3.5-0.8B          # Show noise report
  weightnoise inspect Qwen/Qwen3.5-0.8B --layer 5 # Single layer detail
  weightnoise prune Qwen/Qwen3.5-0.8B --keep 0.5 # Remove 50% of noise
  weightnoise compress Teacher Student --save ./wit-model  # WIT cross-arch transfer
  weightnoise compress Teacher Student --method theseus --calibrate --upload KiriLabs/Model-Name
"""

__version__ = "0.5.0"
