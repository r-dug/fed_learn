"""Experiment configuration and path helpers."""

import os
from dataclasses import dataclass, fields
from typing import Optional

from fedlearn.constants import OUTPUT_DIR_PREFIX


@dataclass
class ExperimentConfig:
    seed: int = 733
    dataset: str = 'mnist'
    model: str = 'cnn'
    classes: int = 10
    batchsize: int = 32
    lr: float = 0.001
    bias: float = 0.5
    momentum: float = 0.0
    gpu: int = 0
    epochs: int = 500
    local_round: int = 1
    nworkers: int = 10
    nbyz: int = 2
    byz_type: str = 'none'
    aggregation: str = 'mean'
    perturbation: str = 'sgn'
    interval: int = 10
    log: str = 'log.txt'
    checkpoint: Optional[str] = None
    quiet: bool = False
    progress_pos: int = 0
    pretrain_epochs: int = 0

    @classmethod
    def from_args(cls, args):
        """Create config from argparse namespace."""
        return cls(**{f.name: getattr(args, f.name) for f in fields(cls) if hasattr(args, f.name)})


def build_para_string(cfg):
    """Build the canonical experiment filename (without directory)."""
    return (
        f"{cfg.seed}+{cfg.dataset}+{cfg.model}+"
        f"bias{cfg.bias}+epoch{cfg.epochs}+"
        f"local{cfg.local_round}+lr{cfg.lr}+"
        f"batch{cfg.batchsize}+"
        f"nwork{cfg.nworkers}+nbyz{cfg.nbyz}+"
        f"{cfg.byz_type}+{cfg.aggregation}+{cfg.perturbation}.txt"
    )


def result_dir(cfg):
    """Return the output directory for an experiment."""
    return os.path.join(OUTPUT_DIR_PREFIX, f"default+{cfg.dataset}+byz_type_{cfg.byz_type}")


def result_path(cfg):
    """Return the full result file path for an experiment."""
    return os.path.join(result_dir(cfg), build_para_string(cfg))


def byzantine_log_path(cfg):
    """Return the Byzantine detection log path for an experiment."""
    return os.path.join(OUTPUT_DIR_PREFIX,
        f"b_log_{cfg.seed}_{cfg.dataset}_{cfg.byz_type}_{cfg.aggregation}.txt")
