# Byzantine-Resilient Federated Learning Experiments

TensorFlow implementation for testing the effectiveness of aggregation methods against Byzantine attacks in federated learning.

## Quick Start

```bash
# Set up conda environment
conda env create -f environment.yml
conda activate federated-learning

# Run quick test (CPU, ~10 min)
python run.py --quick --gpu=-1

# Run full experiment suite (GPU recommended)
python run.py --gpu=0
```

## Project Structure

```
tensorflow/
├── train.py                    # CLI entry point for single experiments
├── run.py                      # Parallel experiment runner
├── aggregate_results.py        # Compute mean/std across seeds
├── tabulate.py                 # Generate Excel tables and plots
├── visualize_results.py        # Generate visualization plots
├── environment.yml             # Conda environment definition
├── setup_env.sh                # Environment setup script
└── fedlearn/                   # Core package
    ├── constants.py            # Shared constants (single source of truth)
    ├── config.py               # ExperimentConfig dataclass + path helpers
    ├── models.py               # CNN, MLR, ResNet model definitions
    ├── data.py                 # Dataset loading (with file locking), worker assignment, triggers
    ├── evaluate.py             # Accuracy evaluation + backdoor ASR measurement
    ├── training.py             # GPU setup, seeding, warmup, training loop
    ├── attacks/                # Byzantine attack implementations
    │   ├── core.py             # no_byz, gaussian, scale
    │   ├── optimization.py     # trim_attack, krum_attack, MinMax, MinSum, lie, IPM
    │   ├── backdoor.py         # model_replacement, model_replacement_adaptive
    │   └── _krum_helpers.py    # Shared Krum scoring (used by attacks + aggregation)
    ├── aggregation/            # Aggregation method implementations
    │   ├── basic.py            # mean, trim, median, krum
    │   └── hdbscan_median.py   # newMedian (GPU-accelerated via cuML)
    └── analysis/               # Results loading and parsing
        └── results_io.py       # Shared parse_filename(), load_all_results()
```

## Experiment Configuration

| Parameter | Values |
|-----------|--------|
| Seeds | 5 repetitions (0-4) |
| Datasets | MNIST, FashionMNIST, CIFAR-10 |
| Models | CNN, MLR, ResNet |
| Workers | 1000 total, 1 Byzantine |
| Epochs | 2500 (full) / 1 (quick) |

### Byzantine Attacks

| Name | Type | Description |
|------|------|-------------|
| `none` | Baseline | No attack |
| `gauss` | Untargeted | Gaussian noise injection |
| `label` | Untargeted | Label flipping |
| `trimAtt` | Untargeted | Trimmed mean attack |
| `krumAtt` | Untargeted | Krum-targeted attack |
| `scale` | Backdoor | Gradient scaling with trigger |
| `MinMax` | Untargeted | Minimize maximum distance to benign gradients |
| `MinSum` | Untargeted | Minimize sum of distances to benign gradients |
| `lie` | Untargeted | Little Is Enough attack |
| `modelReplace` | Backdoor | Model replacement (Bagdasaryan et al., 2020) |
| `modelReplaceAdapt` | Backdoor | Adaptive model replacement with evasion |
| `IPM` | Untargeted | Inner product manipulation |

### Aggregation Methods

| Name | Description |
|------|-------------|
| `mean` | Simple averaging |
| `trim` | Coordinate-wise trimmed mean |
| `median` | Coordinate-wise median |
| `krum` | Krum selection |
| `newMedian` | HDBSCAN-based robust median (novel method, GPU-accelerated) |

## Usage

### Run Experiments

```bash
# Show configuration without running
python run.py --dry-run

# Quick test (1 seed, 1 epoch)
python run.py --quick --gpu=-1

# Full suite (5 seeds, 2500 epochs)
python run.py --gpu=0

# Resume after interruption (skip completed experiments)
python run.py --resume --gpu=0
```

### Experiment Runner Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--quick` | off | Quick test mode |
| `--threads` | 0 (auto) | Number of parallel experiments |
| `--gpu` | 0 | GPU index (-1 for CPU) |
| `--dry-run` | off | Print configuration without running |
| `--target-util` | 0.85 | Target GPU memory utilization (0.0-1.0) |
| `--mem-per-exp` | 0 (auto) | Estimated memory per experiment in MB |
| `--resume` | off | Skip experiments with existing output |
| `--oom-retries` | 1 | Retry count after CUDA OOM |

### Single Experiment

```bash
python train.py \
    --dataset=mnist \
    --model=cnn \
    --epochs=100 \
    --lr=0.01 \
    --nworkers=1000 \
    --nbyz=1 \
    --byz_type=gauss \
    --aggregation=newMedian \
    --gpu=0
```

#### Checkpoint Support

```bash
# Run base case (auto-saves checkpoint)
python train.py --dataset=mnist --model=cnn --byz_type=none --seed=0 --aggregation=mean

# Load checkpoint for transfer learning (skips warmup)
python train.py --dataset=mnist --model=cnn --byz_type=gauss --aggregation=newMedian \
    --checkpoint=out/checkpoints/mnist_cnn.weights.h5
```

### Analyze Results

```bash
# Compute mean +/- std across seeds
python aggregate_results.py --save

# Generate Excel tables
python tabulate.py --task=tabulate

# Generate training curve plots
python tabulate.py --task=graph_error

# Generate all visualizations
python visualize_results.py

# Specific visualization tasks
python visualize_results.py --task=curves     # Training curves
python visualize_results.py --task=heatmap    # Accuracy heatmaps
python visualize_results.py --task=ranking    # Aggregation ranking
```

## Output

Results are saved to:
```
out/default+{dataset}+byz_type_{attack}/{seed}+{dataset}+{model}+...+{aggregation}+{perturbation}.txt
```

Each file contains accuracy values per evaluation interval. For backdoor attacks (`scale`, `modelReplace`, `modelReplaceAdapt`), two columns: clean accuracy and attack success rate (ASR).

## Environment

Requires conda with CUDA support:

```bash
conda env create -f environment.yml
conda activate federated-learning
```

Key dependencies: Python 3.12, TensorFlow 2.19, cuML 26.02 (RAPIDS), scikit-learn, NumPy, SciPy, pandas, matplotlib, tqdm.

GPU-accelerated HDBSCAN uses cuML when available; falls back to sklearn on CPU.

## Citation

If using the `newMedian` aggregation method, please cite the associated paper.
