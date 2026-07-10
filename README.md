# weightnoise

**Visualize, prune, and transfer intelligence across neural network weights.**

`weightnoise` analyzes every weight matrix in a transformer to distinguish signal from noise, prune noise safely, and compress large teacher models into small student architectures via WIT (Weight-Space Intelligence Transfer).

## Install

```bash
pip install weightnoise
```

Requires PyTorch 2.0+. Works on CPU for models up to 4B params.

## Commands

### `inspect` — Noise report for any transformer

```bash
# Full noise profile
weightnoise inspect Qwen/Qwen3.5-0.8B

# Single layer detail
weightnoise inspect Qwen/Qwen3.5-0.8B --layer 5
```

Outputs per-matrix noise statistics: effective rank, spectral concentration, Wanda importance, kurtosis, and adaptive noise percentage.

### `prune` — Remove noise without losing quality

```bash
weightnoise prune Qwen/Qwen3.5-0.8B --keep 0.9 --method wanda
weightnoise prune Qwen/Qwen3.5-0.8B --keep 0.5 --method spectral
```

Validated: 5-10% Wanda pruning *improves* perplexity on distilgpt2 (confirmed noise removal).

### `plan` — Hardware-aware compression plan

```bash
weightnoise plan Qwen/Qwen3.5-0.8B
```

Recommends prune/quantization settings based on your hardware profile.

### `compress` — Cross-architecture WIT transfer **[NEW in v0.5.0]**

```bash
# SVD spectral stitching (works without data)
weightnoise compress Qwen/Qwen3.5-4B Qwen/Qwen3.5-0.8B --save ./compressed

# Theseus Procrustes alignment (uses activation data, ICML 2026)
weightnoise compress Qwen/Qwen3.5-4B Qwen/Qwen3.5-0.8B \
  --method theseus --calibrate --save ./compressed

# Full pipeline: compress + auto-publish to HuggingFace
weightnoise compress Qwen/Qwen3.6-27B Qwen/Qwen3.5-0.8B \
  --method theseus --calibrate \
  --upload KiriLabs/WIT-Model-Name
```

Two methods:
- **`svd`** (default): mean-averaging + SVD projection. Preserves spectral structure without needing any calibration data. The teacher weights are mean-averaged across corresponding layers, then SVD-projected to the student's exact dimensions.
- **`theseus`**: Procrustes alignment from activation cross-covariance. Based on Theseus (Salici et al., ICML 2026). Runs calibration data through both models to learn optimal linear maps between teacher and student representational spaces. Transport: `W_s = T_out @ W_t @ T_in^T`.

Streaming support for large teachers (100B+):

```bash
weightnoise compress Qwen/Qwen3.6-27B Qwen/Qwen3.5-0.8B \
  --stream --save ./compressed
```

## How Noise Is Measured

For each weight matrix, three independent metrics. All thresholds are adaptive.

**1. Adaptive Wanda Importance** (`|w| × column_norm`)  
Per output neuron, importance = weight magnitude × input activation norm. Noise floor = 5th percentile of per-row relative scores.

**2. Spectral Analysis (SVD)**  
- Effective rank: Renyi entropy of singular values  
- Concentration ratio: % energy in top 10% of singular values  
- Rank retention: rank needed for 90/95/99% energy

**3. Distribution Analysis**  
- Kurtosis: heavy tails = structured features, ~3 = noise  
- KL divergence from Gaussian  
- 2xMAD thresholding (robust magnitude detection)

## Architecture Support

| Model Family | Pattern | Status |
|-------------|---------|--------|
| GPT-2 | `transformer.h.N` | ✅ |
| LLaMA / Mistral | `model.layers.N` | ✅ |
| Qwen3.5/3.6 | `model.layers.N` | ✅ (nested config) |
| Gemma 4 | `model.layers.N` | ✅ |
| BERT | `encoder.layer.N` | ✅ |
| T5 / Flan-T5 | `decoder.layer.N` | ✅ |

## Design

- **CPU-first**: inspection and compression run on CPU. No GPU needed.
- **Data-adaptive thresholds**: computed from the model's own weight distribution, no magic constants.
- **No fine-tuning**: composition-only. If the transfer doesn't work, we document it and improve the method.
- **Cross-architecture**: any teacher → any student, regardless of family. Architecture gap = research problem, not blocker.

## References

- Theseus: Salici et al., "Cross-Architecture Weight Transfer via Optimal Transport", ICML 2026
- Wanda: Sun et al., "A Simple and Effective Pruning Approach for Large Language Models" (2024)
- SparseGPT: Frantar & Alistarh, "Massive Language Models Can Be Accurately Pruned in One-Shot" (2023)

## License

MIT
