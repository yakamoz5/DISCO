# DISCO: Mitigating Bias in Deep Learning with Conditional Distance Correlation

Official implementation of **DISCO** (**DIS**tance **CO**rrelation for Conditional Independence) for causal stability and bias mitigation in deep learning models (ICML 2026).

[![arXiv](https://img.shields.io/badge/arXiv-2506.1165-b31b1b.svg)](https://arxiv.org/abs/2506.11653)

---

## Important Notice & Core Contribution

The primary algorithmic contribution of this repository resides in the [**`utils/`**] directory, specifically [**`utils/cdisco_torch.py`**]:

- **Single-Shot DISCO (`sdisco_metric`)**: An exact algebraic factorization of conditional distance correlation that reduces the memory requirement from prohibitive $\mathcal{O}(n^3)$ tensor allocations to a scalable $\mathcal{O}(n^2)$ footprint, enabling efficient backpropagation in deep neural networks.
- **Gradient Optimization Design**: In alignment with the paper's formulation $\min_\theta \mathcal{L}(Y, g_\theta(X)) + \lambda \cdot \text{sDISCO}(\hat{Y}, B \mid Y)$, the implementation in [**`utils/cdisco_torch.py`**] is strictly optimized for training:
  - Gradients flow **only** through the first argument `X` (the model predictions $\hat{Y}$).
  - The second argument `Y` (bias attribute $B$) and third argument `Z` (conditioning target $Y$) are computed inside `torch.no_grad()` blocks to minimize memory overhead and peak GPU utilization.
- **Custom Usage & Metric Testing**:
  - **Optimizing other inputs**: If your application requires differentiating with respect to the bias variable or conditioning label, modify the `with torch.no_grad():` block in [**`utils/cdisco_torch.py`**].
  - **Statistical Testing & Offline Evaluation**: When utilizing `sdisco_metric` purely as an evaluation metric, independence test, or dependency measure (without loss backpropagation), wrap your call in `torch.no_grad()`.

```python
import torch
from utils.cdisco_torch import sdisco_metric

# Example: Differentiable loss during training
y_hat = model(images)      # Predictions (requires_grad=True)
bias = batch["bias"]       # Protected attribute (detached)
target = batch["target"]   # Ground truth label (detached)

loss_disco = sdisco_metric(X=y_hat, Y=bias, Z=target, h=1.0, method="max")

# Example: Statistical evaluation only
with torch.no_grad():
    dep_score = sdisco_metric(X=features, Y=bias, Z=target, h=1.0)
```

---

## Project Structure

The repository is modularized into the standalone core utility package and the experimental benchmarking framework:

```
├── utils/                                # Core installable package
│   ├── __init__.py                       # Package exports (sdisco_metric, callbacks)
│   ├── cdisco_torch.py                   # Differentiable sDISCO estimator
│   └── callbacks.py                      # Lightning timing and WandB callbacks
├── ICML/                                 # Experimental suite & benchmarks
│   ├── debiasing_methods.py              # LightningModules (cDiscoPredictor, MetaDataPrediction)
│   ├── resnet.py / resnet_gn.py          # Backbones (ResNet-18, ResNet-50, GroupNorm ResNet, ConvNet)
│   ├── train.py                          # Hydra training entrypoint
│   ├── train_test_pipeline.py            # Train + test pipeline with checkpoint evaluation
│   ├── config/                           # Hierarchical Hydra configurations
│   │   ├── standard.yaml                 # Default training entry config
│   │   ├── standard_test.yaml            # Default train + test pipeline config
│   │   ├── timing_analysis.yaml          # Timing benchmark config
│   │   ├── paths/                        # Base data paths
│   │   ├── base/                         # Dataset configurations (train)
│   │   ├── base_test/                    # Dataset configurations (test)
│   │   ├── experiment/                   # Model & debiasing sweeps
│   │   └── experiment_test/              # Test-specific experiment configs
│   ├── data/                             # Dataset classes & LightningDataModules
│   │   ├── dSprites/                     # Synthetic 2D geometry shape benchmark
│   │   ├── fairface/                     # Real face demographic bias benchmark
│   │   ├── mnli/                         # NLP TinyBERT negation bias benchmark
│   │   ├── sim_data/                     # Synthetic 2-Gaussian-blob benchmark
│   │   ├── waterbirds/                   # Waterbirds spurious background benchmark
│   │   └── yaleB/                        # Extended YaleB face pose & lighting benchmark
│   └── scripts/                          # Execution and sweep scripts
│       ├── run_cdisco_exps.sh            # Run experiments across datasets
│       └── run_timing_analysis.sh        # Run timing benchmarks
└── pyproject.toml                        # Build configuration (cdisco v0.1.0)
```

---

## Configuration with Hydra (YAML)

All training runs and evaluation pipelines are controlled through hierarchical YAML configurations managed by [Hydra](https://hydra.cc/):

- [**`config/paths/paths.yaml`**]: Sets the root data directory (`dataset_path: /workspace/data`).
- [**`config/base/<dataset>.yaml`**]: Defines dataset-specific settings, including DataLoader worker counts, batch sizes, target type (`label`, `label_cat`, `label_c`), backbone encoder (`resnet_18`, `resnet_gn`, `tinybert`), loss configurations, learning rate, and WandB logging settings.
- [**`config/experiment/<method>.yaml`**]: Configures the debiasing method (`cdisco` for sDISCO, `resnet` for unregularized baseline) along with hyperparameter search grids (bandwidth `bw`, penalty strength `cdcor_lambda`, `warmup_ratio`, and random seeds).

---

## Dataset Architecture & Directory Layout

The data modules expect datasets structured under `${dataset_path}` (configured in `config/paths/paths.yaml`, default: `/workspace/data/`):

```
/workspace/data/
├── yaleB/
│   └── ExtendedYaleB/
│       ├── _base_splits/
│       │   ├── train_partitioned_biased.csv
│       │   ├── val_partitioned.csv
│       │   └── test_partitioned.csv
│       └── <subject_directories>/        # e.g., yaleB01/, yaleB02/
├── waterbirds_prediction/
│   ├── data/                             # Cropped bird images
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── fairface/
│   ├── img/                              # Face images
│   ├── train_biased.csv
│   ├── val.csv
│   └── test.csv
├── dSprites/
│   ├── sprites.h5                        # HDF5 containing 'images' dataset (N, 64, 64)
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── sim_data/
│   ├── sim_biased.h5                     # HDF5 groups with 'img', 'label_c', 'cf_std'
│   └── split_biased.json                 # JSON dictionary with train/val/test ID lists
└── MNLI/
    ├── processed_splits/                 # CSV files with 'gold_label', 'sentence2_has_negation'
    └── processed_features/               # Pre-tokenized .pt tensor files for TinyBERT
```

### Dataset Schema Summary

| Dataset | Modality / Backbone | Target ($Y$) | Target Type | Bias ($B$) | Split / Annotation Files |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **YaleB** | Vision / ResNet-18 | Face Pose (3 classes) | `label_cat` | Normalized Azimuth & Elevation | `train_partitioned_biased.csv`, `val_partitioned.csv`, `test_partitioned.csv` (`file`, `pose_category`, `azimuth`, `elevation`) |
| **Waterbirds** | Vision / ResNet-18 | Bird Type (Land vs. Water) | `label` | Background (Land vs. Water) | `train.csv`, `val.csv`, `test.csv` (`unique_img_filename`, `y`, `place`) |
| **FairFace** | Vision / ResNet-18 | Labeled Sex (Male vs. Female) | `label` | Skin Tone (Light vs. Dark) | `train_biased.csv`, `val.csv`, `test.csv` (`file`, `gender`, `bin_ethnicity`) |
| **dSprites** | Vision / ResNet-18 | Y-Position (Continuous) | `label_c` | X-Position (Confounder) | `sprites.h5`, `train.csv`, `val.csv`, `test.csv` (`index`, `x`, `y`, `scale`, `shape`) |
| **Blob Simulation**| Vision / ResNet-GN | Causal Blob Intensity | `label_c` | Spurious Blob Intensity | `sim_biased.h5` + `split_biased.json` (`img`, `label_c`, `cf_std`) |
| **MNLI** | NLP / TinyBERT | Entailment (3 classes) | `label_cat` | Negation Token Presence | `processed_splits/*.csv`, `processed_features/*.pt` (`gold_label`, `sentence2_has_negation`) |

---

## Installation

Install the repository as an editable package:

```bash
git clone https://github.com/yakamoz5/DISCO.git
cd DISCO
pip install -e .
```

---

## Usage & Execution

### 1. Training with Hydra

Run individual experiments or sweeps across datasets and hyperparameters:

```bash
# Train sDISCO on YaleB with default parameters
python ICML/train.py base@_global_=yaleB experiment@_global_=cdisco

# Run unregularized ResNet baseline on Waterbirds
python ICML/train.py base@_global_=waterbirds experiment@_global_=resnet

# Run hyperparameter grid sweep using Hydra multirun (-m)
python ICML/train.py -m base@_global_=waterbirds experiment@_global_=cdisco
```

### 2. End-to-End Train and Test Pipeline

To train and automatically evaluate on the unbiased test set using the best validation checkpoint:

```bash
python ICML/train_test_pipeline.py base@_global_=dsprites experiment_test@_global_=cdisco best_monitor=mse
```

### 3. Batch Scripts

Shell scripts under [**`ICML/scripts/`**] facilitate multi-dataset execution and benchmarks:

```bash
# Execute sDISCO experiments
bash ICML/scripts/run_cdisco_exps.sh

# Run epoch timing and memory analysis
bash ICML/scripts/run_timing_analysis.sh
```

---

## Citation

If you find this work or the sDISCO estimator useful in your research, please cite:

```bibtex
@inproceedings{kavak2026disco,
  title     = {{DISCO}: Mitigating Bias in Deep Learning with Conditional Distance Correlation},
  author    = {Kavak, Emre and Wolf, Tom Nuno and Wachinger, Christian},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026}
}
```

