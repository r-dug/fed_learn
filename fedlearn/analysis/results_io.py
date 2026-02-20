"""Shared I/O helpers for loading experiment results."""

import os
import numpy as np
from collections import defaultdict

from fedlearn.constants import BACKDOOR_ATTACKS


def parse_filename(filename):
    """Parse experiment parameters from filename.

    Format: seed+dataset+model+bias0.5+epoch2550+local10+lr0.01+batch32+nwork100+nbyz20+byz_type+aggregation+perturbation.txt
    """
    parts = filename.replace('.txt', '').split('+')
    if len(parts) < 13:
        return None

    try:
        return {
            'seed': int(parts[0]),
            'dataset': parts[1],
            'model': parts[2],
            'bias': parts[3].replace('bias', ''),
            'epochs': parts[4].replace('epoch', ''),
            'local_round': parts[5].replace('local', ''),
            'lr': parts[6].replace('lr', ''),
            'batchsize': parts[7].replace('batch', ''),
            'nworkers': parts[8].replace('nwork', ''),
            'nbyz': parts[9].replace('nbyz', ''),
            'byz_type': parts[10],
            'aggregation': parts[11],
            'perturbation': parts[12],
        }
    except (IndexError, ValueError):
        return None


def load_all_results(out_dir='out'):
    """Load all experiment results with full time series data.

    Returns:
        dict mapping (dataset, model, byz_type) -> {aggregation: [run_dicts]}
        Each run_dict has keys: seed, accuracy, loss, backdoor (or None).
    """
    results = defaultdict(lambda: defaultdict(list))

    if not os.path.isdir(out_dir):
        return results

    for subdir in os.listdir(out_dir):
        subdir_path = os.path.join(out_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        for filename in os.listdir(subdir_path):
            if not filename.endswith('.txt'):
                continue

            params = parse_filename(filename)
            if params is None:
                continue

            filepath = os.path.join(subdir_path, filename)
            try:
                data = np.loadtxt(filepath)
                if data.ndim == 0:
                    data = np.array([data])

                if params['byz_type'] in BACKDOOR_ATTACKS and data.ndim == 2:
                    accuracy = data[:, 0]
                    backdoor = data[:, 1] if data.shape[1] > 1 else None
                else:
                    accuracy = data if data.ndim == 1 else data[:, 0]
                    backdoor = None

                key = (params['dataset'], params['model'], params['byz_type'])
                results[key][params['aggregation']].append({
                    'seed': params['seed'],
                    'accuracy': accuracy,
                    'loss': 1 - accuracy,
                    'backdoor': backdoor,
                })
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")

    return results
