#!/usr/bin/env python3
"""
Aggregate results from multiple experiment runs (different seeds).

This script reads results from the out/ directory, computes mean and std
across different seeds, and outputs summary statistics.
"""

import os
import numpy as np
import pandas as pd
import argparse
from collections import defaultdict
import warnings

from fedlearn.constants import BACKDOOR_ATTACKS

warnings.filterwarnings('ignore')


def parse_filename(filename):
    """Parse experiment parameters from filename."""
    parts = filename.replace('.txt', '').split('+')
    if len(parts) < 13:
        return None

    try:
        params = {
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
            'perturbation': parts[12]
        }
        return params
    except (IndexError, ValueError):
        return None


def get_experiment_key(params):
    """Get a key that identifies an experiment configuration (excluding seed)."""
    return (
        params['dataset'],
        params['model'],
        params['byz_type'],
        params['aggregation']
    )


def load_results(out_dir='out'):
    """Load all results from output directories."""
    results = defaultdict(list)

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

                # Handle backdoor attacks (have 2 columns: accuracy and backdoor_acc)
                if params['byz_type'] in BACKDOOR_ATTACKS and data.ndim == 2:
                    final_acc = data[-1, 0] if len(data.shape) > 1 else data[-1]
                    final_backdoor = data[-1, 1] if data.shape[1] > 1 else None
                    results[get_experiment_key(params)].append({
                        'seed': params['seed'],
                        'final_acc': final_acc,
                        'final_backdoor': final_backdoor,
                        'all_acc': data[:, 0] if data.ndim == 2 else data
                    })
                else:
                    final_acc = data[-1] if data.ndim == 1 else data[-1, 0]
                    results[get_experiment_key(params)].append({
                        'seed': params['seed'],
                        'final_acc': final_acc,
                        'all_acc': data if data.ndim == 1 else data[:, 0]
                    })
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")

    return results


def compute_statistics(results):
    """Compute mean and std across seeds for each experiment configuration."""
    stats = []

    for key, runs in results.items():
        dataset, model, byz_type, aggregation = key

        final_accs = [r['final_acc'] for r in runs]
        n_runs = len(runs)

        stat = {
            'dataset': dataset,
            'model': model,
            'byz_type': byz_type,
            'aggregation': aggregation,
            'n_runs': n_runs,
            'mean_acc': np.mean(final_accs),
            'std_acc': np.std(final_accs),
            'min_acc': np.min(final_accs),
            'max_acc': np.max(final_accs)
        }

        # Add backdoor stats for backdoor attacks
        if byz_type in BACKDOOR_ATTACKS and 'final_backdoor' in runs[0]:
            backdoor_accs = [r['final_backdoor'] for r in runs if r['final_backdoor'] is not None]
            if backdoor_accs:
                stat['mean_backdoor'] = np.mean(backdoor_accs)
                stat['std_backdoor'] = np.std(backdoor_accs)

        stats.append(stat)

    return pd.DataFrame(stats)


def create_summary_tables(df):
    """Create pivot tables for easy comparison, including ASR for backdoor attacks."""
    tables = {}

    for dataset in df['dataset'].unique():
        for model in df['model'].unique():
            subset = df[(df['dataset'] == dataset) & (df['model'] == model)]
            if len(subset) == 0:
                continue

            # Create pivot table: rows=byz_type, cols=aggregation, values=mean_acc
            pivot = subset.pivot_table(
                index='byz_type',
                columns='aggregation',
                values='mean_acc',
                aggfunc='first'
            )

            # Also create std table
            pivot_std = subset.pivot_table(
                index='byz_type',
                columns='aggregation',
                values='std_acc',
                aggfunc='first'
            )

            table_entry = {
                'mean': pivot,
                'std': pivot_std,
                'n_runs': subset['n_runs'].iloc[0] if len(subset) > 0 else 0
            }

            # ASR pivot tables for backdoor attacks
            backdoor_subset = subset[subset['byz_type'].isin(BACKDOOR_ATTACKS)]
            if len(backdoor_subset) > 0 and 'mean_backdoor' in backdoor_subset.columns:
                asr_subset = backdoor_subset.dropna(subset=['mean_backdoor'])
                if len(asr_subset) > 0:
                    table_entry['mean_asr'] = asr_subset.pivot_table(
                        index='byz_type',
                        columns='aggregation',
                        values='mean_backdoor',
                        aggfunc='first'
                    )
                    table_entry['std_asr'] = asr_subset.pivot_table(
                        index='byz_type',
                        columns='aggregation',
                        values='std_backdoor',
                        aggfunc='first'
                    )

            tables[f'{dataset}_{model}'] = table_entry

    return tables


def main():
    parser = argparse.ArgumentParser(description='Aggregate experiment results across seeds')
    parser.add_argument('--out_dir', type=str, default='out', help='Output directory')
    parser.add_argument('--save', action='store_true', help='Save results to CSV')
    args = parser.parse_args()

    print("=" * 70)
    print("AGGREGATING RESULTS FROM MULTIPLE RUNS")
    print("=" * 70)

    # Load results
    results = load_results(args.out_dir)

    if not results:
        print("No results found!")
        return

    print(f"Found {len(results)} unique experiment configurations")

    # Compute statistics
    df = compute_statistics(results)

    # Create summary tables
    tables = create_summary_tables(df)

    # Print results
    for name, table_data in tables.items():
        print(f"\n{'=' * 70}")
        print(f"RESULTS: {name} (n={table_data['n_runs']} runs)")
        print("=" * 70)

        print("\nMean Accuracy:")
        print(table_data['mean'].round(4).to_string())

        print("\nStd Accuracy:")
        print(table_data['std'].round(4).to_string())

        if 'mean_asr' in table_data:
            print("\nMean ASR (Backdoor Attack Success Rate — lower is better):")
            print(table_data['mean_asr'].round(4).to_string())

            print("\nStd ASR:")
            print(table_data['std_asr'].round(4).to_string())

    # Save to CSV if requested
    if args.save:
        df.to_csv(os.path.join(args.out_dir, 'aggregated_results.csv'), index=False)
        print(f"\nSaved aggregated results to {args.out_dir}/aggregated_results.csv")

        # Save individual tables
        for name, table_data in tables.items():
            table_data['mean'].to_csv(os.path.join(args.out_dir, f'{name}_mean.csv'))
            table_data['std'].to_csv(os.path.join(args.out_dir, f'{name}_std.csv'))
            if 'mean_asr' in table_data:
                table_data['mean_asr'].to_csv(os.path.join(args.out_dir, f'{name}_mean_asr.csv'))
                table_data['std_asr'].to_csv(os.path.join(args.out_dir, f'{name}_std_asr.csv'))
        print(f"Saved individual tables to {args.out_dir}/")

    # Print overall summary
    print(f"\n{'=' * 70}")
    print("OVERALL SUMMARY")
    print("=" * 70)

    # Best aggregation for each attack type
    for byz_type in df['byz_type'].unique():
        subset = df[df['byz_type'] == byz_type]
        best = subset.loc[subset['mean_acc'].idxmax()]
        line = f"{byz_type:20s}: Best acc = {best['aggregation']:10s} ({best['mean_acc']:.4f} +/- {best['std_acc']:.4f})"

        # For backdoor attacks, also show best (lowest) ASR
        if byz_type in BACKDOOR_ATTACKS and 'mean_backdoor' in subset.columns:
            asr_subset = subset.dropna(subset=['mean_backdoor'])
            if len(asr_subset) > 0:
                best_asr = asr_subset.loc[asr_subset['mean_backdoor'].idxmin()]
                line += f" | Best ASR = {best_asr['aggregation']:10s} ({best_asr['mean_backdoor']:.4f})"

        print(line)


if __name__ == '__main__':
    main()
