#!/bin/bash
#
# Setup script for Federated Learning experiments with TensorFlow
# Uses conda (Anaconda/Miniforge) for environment and CUDA management
#
# Usage:
#   ./setup_env.sh          # Full setup (interactive)
#   ./setup_env.sh --batch  # Non-interactive setup (for cluster batch jobs)
#   ./setup_env.sh --activate-only  # Just activate existing env
#   source setup_env.sh     # Setup and keep environment active in current shell
#

set -e  # Exit on error

# Non-interactive mode for batch jobs
BATCH_MODE=false
for arg in "$@"; do
    if [[ "$arg" == "--batch" ]]; then
        BATCH_MODE=true
    fi
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_NAME="federated-learning"
ENV_FILE="$SCRIPT_DIR/environment.yml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_header() {
    echo ""
    echo "========================================"
    echo "$1"
    echo "========================================"
}

# =============================================================================
# Check for active venv — bail early to prevent environment conflicts
# =============================================================================
if [[ -n "$VIRTUAL_ENV" ]]; then
    print_error "A Python virtual environment is currently active: $VIRTUAL_ENV"
    print_error "Deactivate it first with 'deactivate' before running this script."
    print_error "Mixing venv and conda environments can cause unexpected conflicts."
    return 1 2>/dev/null || exit 1
fi

# =============================================================================
# Find conda
# =============================================================================
find_conda() {
    if command -v conda &> /dev/null; then
        return 0
    fi
    # Check common install locations
    for candidate in "$HOME/anaconda3" "$HOME/miniconda3" "$HOME/miniforge3" "$HOME/mambaforge"; do
        if [[ -f "$candidate/bin/conda" ]]; then
            eval "$("$candidate/bin/conda" shell.bash hook)"
            return 0
        fi
    done
    return 1
}

if ! find_conda; then
    print_error "conda not found. Install Miniforge from: https://github.com/conda-forge/miniforge"
    exit 1
fi

CONDA_BASE=$(conda info --base)
print_status "Found conda at: $CONDA_BASE ($(conda --version))"

# =============================================================================
# Check for --activate-only flag
# =============================================================================
if [[ "$1" == "--activate-only" ]] || [[ "$2" == "--activate-only" ]]; then
    if conda env list | grep -q "^${ENV_NAME} "; then
        conda activate "$ENV_NAME"
        print_status "Environment activated: $ENV_NAME"
        cd "$SCRIPT_DIR"
        print_status "Working directory: $SCRIPT_DIR"
    else
        print_error "Conda environment '$ENV_NAME' not found"
        print_error "Run without --activate-only to create it"
        exit 1
    fi
    return 0 2>/dev/null || exit 0
fi

print_header "Federated Learning Environment Setup (conda)"

# =============================================================================
# Check system requirements
# =============================================================================
print_header "Checking System Requirements"

# Check GPU/driver status (informational only — conda manages CUDA toolkit)
if command -v nvidia-smi &> /dev/null; then
    NVIDIA_DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    CUDA_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
    if [[ -n "$NVIDIA_DRIVER" ]]; then
        print_status "NVIDIA Driver: $NVIDIA_DRIVER"
        print_status "GPU: $GPU_NAME (compute capability $CUDA_CAP)"
        HAS_GPU=true
    else
        HAS_GPU=false
    fi
else
    print_warning "No NVIDIA GPU detected or drivers not installed"
    print_warning "TensorFlow will run on CPU only, cuML will not be available"
    HAS_GPU=false

    if [[ "$BATCH_MODE" != true ]]; then
        read -p "Continue with CPU-only setup? [Y/n] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Nn]$ ]]; then
            print_error "Setup cancelled"
            exit 1
        fi
    fi
fi

# =============================================================================
# Setup Conda Environment
# =============================================================================
print_header "Setting Up Conda Environment"

if conda env list | grep -q "^${ENV_NAME} "; then
    print_status "Conda environment '$ENV_NAME' already exists"
    if [[ "$BATCH_MODE" == true ]]; then
        print_status "Batch mode: updating existing environment"
        conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
    else
        read -p "Recreate environment from scratch? [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Removing old environment..."
            conda deactivate 2>/dev/null || true
            conda env remove -n "$ENV_NAME" -y
            print_status "Creating environment from $ENV_FILE..."
            conda env create -f "$ENV_FILE"
        else
            read -p "Update existing environment? [Y/n] " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Nn]$ ]]; then
                print_status "Updating environment..."
                conda env update -n "$ENV_NAME" -f "$ENV_FILE" --prune
            fi
        fi
    fi
else
    print_status "Creating conda environment '$ENV_NAME' from $ENV_FILE..."
    conda env create -f "$ENV_FILE"
fi

# Activate environment
conda activate "$ENV_NAME"
print_status "Environment '$ENV_NAME' activated"

# =============================================================================
# Verify Installation
# =============================================================================
print_header "Verifying Installation"

# Test TensorFlow
print_status "Testing TensorFlow installation..."
python -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f'GPUs available: {len(gpus)}')
    for gpu in gpus:
        print(f'  - {gpu}')
else:
    print('No GPU available - running on CPU')
"

# Test cuML (GPU only)
if [[ "$HAS_GPU" == true ]]; then
    print_status "Testing cuML installation..."
    python -c "
import cuml
print(f'cuML version: {cuml.__version__}')
" || print_warning "cuML import failed — GPU may not be accessible"
fi

# Test other imports
print_status "Testing other dependencies..."
python -c "
import numpy as np
import scipy
import sklearn
import pandas as pd
import matplotlib
print(f'NumPy: {np.__version__}')
print(f'SciPy: {scipy.__version__}')
print(f'scikit-learn: {sklearn.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Matplotlib: {matplotlib.__version__}')
"

# =============================================================================
# Create output directory
# =============================================================================
print_status "Creating output directory..."
mkdir -p "$SCRIPT_DIR/out"

# =============================================================================
# Final Summary
# =============================================================================
print_header "Setup Complete!"

echo ""
echo "To activate the environment in the future, run:"
echo "  conda activate $ENV_NAME"
echo ""
echo "Or use this script:"
echo "  source $SCRIPT_DIR/setup_env.sh --activate-only"
echo ""
echo "To run experiments:"
echo "  cd $SCRIPT_DIR"
echo "  python run.py --quick --gpu=-1   # Quick test (CPU)"
echo "  python run.py --gpu=0            # Full suite (GPU)"
echo ""

# Change to script directory
cd "$SCRIPT_DIR"
print_status "Working directory: $SCRIPT_DIR"

# Keep environment active if sourced
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    print_status "Environment is now active in your shell"
fi
