### PRO-SOUNDS

Official implementation for:

**"Optimizing Protein Design Through Uncertainty-Weighted Steering of Protein Language Models"**

---

# Repository Structure

```text
.
├── scripts/                # Shell scripts for steering vector extraction and optimization
├── sv_data/                # Sequence data used for steering vector construction
├── sv_extraction/          # Code for generating steering vectors
└── training_surrogate/     # Code for training probabilistic surrogate models
```

---

# Training Surrogate Models

The repository supports training probabilistic surrogate models for predicting protein fitness properties, including:

- Immunogenicity
- Toxicity
- Solubility

Three uncertainty-aware backbones are currently implemented:

- Monte Carlo Dropout (MC Dropout)
- Flexible Evidential Deep Learning (FEDL)
- Evidential Deep Learning (EDL)

Pretrained surrogate models can be downloaded separately (https://drive.google.com/drive/folders/10OX0OFinP4xOHh2fH8_WKhKlvKDOtGTR?usp=share_link).

Code for training surrogate models on custom datasets is provided in:

```text
training_surrogate/
```

---


# Generating Steering Vectors

Sequence datasets used for steering vector construction are located in:

```text
sv_data/
```

The following scripts generate different steering vector variants:

| Method | Script |
|---|---|
| ASPO Baseline | `extract_sv_mean.sh` |
| Precision-Weighted Steering | `extract_sv_precision.sh` |
| Risk-Averse Steering | `extract_sv_risk_averse.sh` |
| Hybrid Precision Fusion | `extract_sv_hybrid_precision.sh` |
| Hybrid Risk-Averse Fusion | `extract_sv_hybrid_risk_averse.sh` |

Generated steering vectors are stored as PyTorch checkpoint files (`.pt`).

Example:

```bash
bash extract_sv_precision.sh
```

---

# Sequence Optimization

Protein sequence optimization using the generated steering vectors can be performed with:

```bash
bash run_steering.sh
```
Optimized sequences will be saved (`.csv`).

---

# Acknowledgements
```
# TODO
```
