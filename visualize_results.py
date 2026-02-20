#!/usr/bin/env python3
"""
Visualization script for experiment results.

Creates loss/accuracy curves comparing aggregation methods across different
Byzantine attack types and datasets.

Usage:
    python visualize_results.py                          # All visualizations
    python visualize_results.py --task curves            # Training curves only
    python visualize_results.py --task summary           # Summary bar charts only
    python visualize_results.py --task heatmap           # Heatmaps only
    python visualize_results.py --dataset mnist          # Specific dataset
    python visualize_results.py --model mlr              # Specific model
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings

from fedlearn.constants import (
    ALL_AGGREGATIONS as AGGREGATIONS,
    ALL_BYZ_TYPES as BYZ_TYPES,
    BACKDOOR_ATTACKS,
    AGGREGATION_COLORS as COLORS,
    AGGREGATION_DISPLAY as AGG_LABELS,
    BYZ_TYPE_DISPLAY as BYZ_LABELS,
)
from fedlearn.analysis.results_io import load_all_results

warnings.filterwarnings('ignore')


def compute_mean_std_curves(runs, key='accuracy'):
    """Compute mean and std curves across multiple runs, handling different lengths."""
    if not runs:
        return None, None

    # Filter runs that have the requested key with non-None data
    valid_runs = [r for r in runs if r.get(key) is not None]
    if not valid_runs:
        return None, None

    max_len = max(len(r[key]) for r in valid_runs)

    # Pad shorter runs with their last value
    padded = []
    for r in valid_runs:
        vals = r[key]
        if len(vals) < max_len:
            vals = np.pad(vals, (0, max_len - len(vals)), mode='edge')
        padded.append(vals)

    padded = np.array(padded)
    return np.mean(padded, axis=0), np.std(padded, axis=0)


def plot_training_curves_per_attack(results, dataset, model, output_dir='plots'):
    """
    Plot accuracy/loss curves for each attack type, comparing all aggregation methods.
    For backdoor attacks, ASR is overlaid as dotted lines on the accuracy panel.
    """
    os.makedirs(output_dir, exist_ok=True)

    for byz_type in BYZ_TYPES:
        key = (dataset, model, byz_type)
        if key not in results:
            continue

        agg_data = results[key]
        if not agg_data:
            continue

        is_backdoor = byz_type in BACKDOOR_ATTACKS
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        ax_acc = axes[0]
        ax_loss = axes[1]

        for agg in AGGREGATIONS:
            if agg not in agg_data:
                continue

            runs = agg_data[agg]
            mean_acc, std_acc = compute_mean_std_curves(runs, key='accuracy')
            if mean_acc is None:
                continue

            mean_loss = 1 - mean_acc
            epochs = np.arange(len(mean_acc))

            # Accuracy plot (solid lines)
            ax_acc.plot(epochs, mean_acc, color=COLORS[agg], label=AGG_LABELS[agg], linewidth=2)
            ax_acc.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                               color=COLORS[agg], alpha=0.2)

            # ASR overlay (dotted lines) for backdoor attacks
            if is_backdoor:
                mean_asr, std_asr = compute_mean_std_curves(runs, key='backdoor')
                if mean_asr is not None:
                    asr_epochs = np.arange(len(mean_asr))
                    ax_acc.plot(asr_epochs, mean_asr, color=COLORS[agg],
                               linestyle=':', linewidth=2, label=f'{AGG_LABELS[agg]} ASR')
                    ax_acc.fill_between(asr_epochs, mean_asr - std_asr, mean_asr + std_asr,
                                       color=COLORS[agg], alpha=0.1)

            # Loss plot
            ax_loss.plot(epochs, mean_loss, color=COLORS[agg], label=AGG_LABELS[agg], linewidth=2)
            ax_loss.fill_between(epochs, mean_loss - std_acc, mean_loss + std_acc,
                                color=COLORS[agg], alpha=0.2)

        ax_acc.set_xlabel('Epoch', fontsize=12)
        ax_acc.set_ylabel('Accuracy', fontsize=12)
        title_suffix = ' (dotted = ASR)' if is_backdoor else ''
        ax_acc.set_title(f'Accuracy - {BYZ_LABELS.get(byz_type, byz_type)}{title_suffix}', fontsize=14)
        ax_acc.legend(loc='lower right', fontsize=9)
        ax_acc.grid(True, alpha=0.3)
        ax_acc.set_ylim([0, 1])

        ax_loss.set_xlabel('Epoch', fontsize=12)
        ax_loss.set_ylabel('Loss (Error Rate)', fontsize=12)
        ax_loss.set_title(f'Loss - {BYZ_LABELS.get(byz_type, byz_type)}', fontsize=14)
        ax_loss.legend(loc='upper right', fontsize=10)
        ax_loss.grid(True, alpha=0.3)
        ax_loss.set_ylim([0, 1])

        fig.suptitle(f'Training Curves: {model.upper()} on {dataset.upper()}', fontsize=16, y=1.02)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f'{model}_{dataset}_{byz_type}_curves.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def plot_combined_attack_grid(results, dataset, model, output_dir='plots'):
    """
    Create a grid of subplots showing accuracy curves for all attacks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Filter attacks that have data
    available_attacks = [b for b in BYZ_TYPES if (dataset, model, b) in results]
    if not available_attacks:
        print(f"No data found for {dataset}/{model}")
        return

    n_attacks = len(available_attacks)
    n_cols = 3
    n_rows = (n_attacks + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    axes = np.array(axes).flatten()

    for idx, byz_type in enumerate(available_attacks):
        ax = axes[idx]
        key = (dataset, model, byz_type)
        agg_data = results[key]
        is_backdoor = byz_type in BACKDOOR_ATTACKS

        for agg in AGGREGATIONS:
            if agg not in agg_data:
                continue

            runs = agg_data[agg]
            mean_acc, std_acc = compute_mean_std_curves(runs, key='accuracy')
            if mean_acc is None:
                continue

            epochs = np.arange(len(mean_acc))
            ax.plot(epochs, mean_acc, color=COLORS[agg], label=AGG_LABELS[agg], linewidth=1.5)
            ax.fill_between(epochs, mean_acc - std_acc, mean_acc + std_acc,
                           color=COLORS[agg], alpha=0.15)

            # ASR as dotted lines for backdoor attacks
            if is_backdoor:
                mean_asr, std_asr = compute_mean_std_curves(runs, key='backdoor')
                if mean_asr is not None:
                    asr_epochs = np.arange(len(mean_asr))
                    ax.plot(asr_epochs, mean_asr, color=COLORS[agg],
                           linestyle=':', linewidth=1.5)

        title = BYZ_LABELS.get(byz_type, byz_type)
        if is_backdoor:
            title += ' (dotted=ASR)'
        ax.set_title(title, fontsize=12)
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Accuracy', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1])

    # Hide empty subplots
    for idx in range(n_attacks, len(axes)):
        axes[idx].set_visible(False)

    # Add legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(AGGREGATIONS),
               fontsize=11, bbox_to_anchor=(0.5, 1.02))

    fig.suptitle(f'Accuracy Curves by Attack Type: {model.upper()} on {dataset.upper()}',
                 fontsize=16, y=1.06)
    plt.tight_layout()

    save_path = os.path.join(output_dir, f'{model}_{dataset}_all_attacks_grid.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_final_accuracy_comparison(results, dataset, model, output_dir='plots'):
    """
    Bar chart comparing final accuracy of each aggregation method per attack.
    For backdoor attacks, also generates an ASR comparison chart.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect final accuracies and ASR
    data = []
    asr_data = []
    for byz_type in BYZ_TYPES:
        key = (dataset, model, byz_type)
        if key not in results:
            continue

        agg_data = results[key]
        for agg in AGGREGATIONS:
            if agg not in agg_data:
                continue

            runs = agg_data[agg]
            final_accs = [r['accuracy'][-1] for r in runs if len(r['accuracy']) > 0]
            if final_accs:
                data.append({
                    'attack': BYZ_LABELS.get(byz_type, byz_type),
                    'aggregation': AGG_LABELS[agg],
                    'mean_acc': np.mean(final_accs),
                    'std_acc': np.std(final_accs)
                })

            # Collect ASR for backdoor attacks
            if byz_type in BACKDOOR_ATTACKS:
                final_asrs = [r['backdoor'][-1] for r in runs
                              if r.get('backdoor') is not None and len(r['backdoor']) > 0]
                if final_asrs:
                    asr_data.append({
                        'attack': BYZ_LABELS.get(byz_type, byz_type),
                        'aggregation': AGG_LABELS[agg],
                        'mean_asr': np.mean(final_asrs),
                        'std_asr': np.std(final_asrs)
                    })

    if not data:
        return

    df = pd.DataFrame(data)

    # Create grouped bar chart for accuracy
    attacks = df['attack'].unique()
    x = np.arange(len(attacks))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, agg in enumerate(AGGREGATIONS):
        agg_label = AGG_LABELS[agg]
        subset = df[df['aggregation'] == agg_label]
        if len(subset) == 0:
            continue

        means = []
        stds = []
        for attack in attacks:
            row = subset[subset['attack'] == attack]
            if len(row) > 0:
                means.append(row['mean_acc'].values[0])
                stds.append(row['std_acc'].values[0])
            else:
                means.append(0)
                stds.append(0)

        ax.bar(x + i * width, means, width, label=agg_label, color=COLORS[agg],
               yerr=stds, capsize=2, error_kw={'linewidth': 1})

    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel('Final Accuracy', fontsize=12)
    ax.set_title(f'Final Accuracy Comparison: {model.upper()} on {dataset.upper()}', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model}_{dataset}_final_accuracy_bars.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # ASR bar chart for backdoor attacks
    if asr_data:
        df_asr = pd.DataFrame(asr_data)
        backdoor_attacks = df_asr['attack'].unique()
        x_bd = np.arange(len(backdoor_attacks))

        fig, ax = plt.subplots(figsize=(10, 6))

        for i, agg in enumerate(AGGREGATIONS):
            agg_label = AGG_LABELS[agg]
            subset = df_asr[df_asr['aggregation'] == agg_label]
            if len(subset) == 0:
                continue

            means = []
            stds = []
            for attack in backdoor_attacks:
                row = subset[subset['attack'] == attack]
                if len(row) > 0:
                    means.append(row['mean_asr'].values[0])
                    stds.append(row['std_asr'].values[0])
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x_bd + i * width, means, width, label=agg_label, color=COLORS[agg],
                   yerr=stds, capsize=2, error_kw={'linewidth': 1})

        ax.set_xlabel('Backdoor Attack Type', fontsize=12)
        ax.set_ylabel('Attack Success Rate (lower is better)', fontsize=12)
        ax.set_title(f'Backdoor ASR Comparison: {model.upper()} on {dataset.upper()}', fontsize=14)
        ax.set_xticks(x_bd + width * 2)
        ax.set_xticklabels(backdoor_attacks, rotation=45, ha='right')
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([0, 1])

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{model}_{dataset}_backdoor_asr_bars.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def plot_heatmap(results, dataset, model, output_dir='plots'):
    """
    Create heatmaps of final accuracies (rows=attacks, cols=aggregations).
    Generates a separate ASR heatmap for backdoor attacks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build accuracy matrix
    matrix = []
    row_labels = []
    # Build ASR matrix (only backdoor attack rows)
    asr_matrix = []
    asr_row_labels = []

    for byz_type in BYZ_TYPES:
        key = (dataset, model, byz_type)
        if key not in results:
            continue

        agg_data = results[key]
        row = []
        asr_row = []
        for agg in AGGREGATIONS:
            if agg in agg_data:
                runs = agg_data[agg]
                final_accs = [r['accuracy'][-1] for r in runs if len(r['accuracy']) > 0]
                row.append(np.mean(final_accs) if final_accs else np.nan)

                if byz_type in BACKDOOR_ATTACKS:
                    final_asrs = [r['backdoor'][-1] for r in runs
                                  if r.get('backdoor') is not None and len(r['backdoor']) > 0]
                    asr_row.append(np.mean(final_asrs) if final_asrs else np.nan)
            else:
                row.append(np.nan)
                if byz_type in BACKDOOR_ATTACKS:
                    asr_row.append(np.nan)

        if not all(np.isnan(row)):
            matrix.append(row)
            row_labels.append(BYZ_LABELS.get(byz_type, byz_type))

        if asr_row and not all(np.isnan(asr_row)):
            asr_matrix.append(asr_row)
            asr_row_labels.append(BYZ_LABELS.get(byz_type, byz_type))

    if not matrix:
        return

    col_labels = [AGG_LABELS[a] for a in AGGREGATIONS]

    # --- Accuracy heatmap ---
    matrix = np.array(matrix)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, fontsize=11)
    ax.set_yticklabels(row_labels, fontsize=11)

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            val = matrix[i, j]
            if not np.isnan(val):
                text_color = 'white' if val < 0.5 else 'black'
                ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                       color=text_color, fontsize=10)

    ax.set_title(f'Final Accuracy Heatmap: {model.upper()} on {dataset.upper()}', fontsize=14)

    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.set_ylabel('Accuracy', rotation=-90, va='bottom', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model}_{dataset}_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

    # --- ASR heatmap (backdoor attacks only) ---
    if asr_matrix:
        asr_matrix = np.array(asr_matrix)
        fig, ax = plt.subplots(figsize=(10, max(4, len(asr_row_labels) + 1)))
        # Reversed colormap: green=low ASR (good defense), red=high ASR (bad defense)
        im = ax.imshow(asr_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(asr_row_labels)))
        ax.set_xticklabels(col_labels, fontsize=11)
        ax.set_yticklabels(asr_row_labels, fontsize=11)

        plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')

        for i in range(len(asr_row_labels)):
            for j in range(len(col_labels)):
                val = asr_matrix[i, j]
                if not np.isnan(val):
                    text_color = 'white' if val > 0.5 else 'black'
                    ax.text(j, i, f'{val:.3f}', ha='center', va='center',
                           color=text_color, fontsize=10)

        ax.set_title(f'Backdoor ASR Heatmap: {model.upper()} on {dataset.upper()} (lower is better)', fontsize=14)

        cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.set_ylabel('Attack Success Rate', rotation=-90, va='bottom', fontsize=11)

        plt.tight_layout()
        save_path = os.path.join(output_dir, f'{model}_{dataset}_asr_heatmap.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")


def plot_aggregation_ranking(results, output_dir='plots'):
    """
    Summary plot showing average ranking of each aggregation method across all attacks.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Collect all final accuracies
    all_data = []
    for key, agg_data in results.items():
        dataset, model, byz_type = key
        for agg in AGGREGATIONS:
            if agg not in agg_data:
                continue
            runs = agg_data[agg]
            final_accs = [r['accuracy'][-1] for r in runs if len(r['accuracy']) > 0]
            if final_accs:
                all_data.append({
                    'dataset': dataset,
                    'model': model,
                    'attack': byz_type,
                    'aggregation': agg,
                    'mean_acc': np.mean(final_accs)
                })

    if not all_data:
        return

    df = pd.DataFrame(all_data)

    # Compute average accuracy per aggregation
    avg_by_agg = df.groupby('aggregation')['mean_acc'].mean().reset_index()
    avg_by_agg = avg_by_agg.sort_values('mean_acc', ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(avg_by_agg)), avg_by_agg['mean_acc'],
                  color=[COLORS[a] for a in avg_by_agg['aggregation']])

    ax.set_xticks(range(len(avg_by_agg)))
    ax.set_xticklabels([AGG_LABELS[a] for a in avg_by_agg['aggregation']], fontsize=12)
    ax.set_ylabel('Average Accuracy', fontsize=12)
    ax.set_title('Average Final Accuracy Across All Attacks and Datasets', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 1])

    # Add value labels on bars
    for bar, val in zip(bars, avg_by_agg['mean_acc']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=11)

    plt.tight_layout()
    save_path = os.path.join(output_dir, 'aggregation_ranking.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_convergence_speed(results, dataset, model, output_dir='plots'):
    """
    Plot showing epochs to reach target accuracy for each method.
    """
    os.makedirs(output_dir, exist_ok=True)

    target_acc = 0.8  # Target accuracy threshold

    data = []
    for byz_type in BYZ_TYPES:
        key = (dataset, model, byz_type)
        if key not in results:
            continue

        agg_data = results[key]
        for agg in AGGREGATIONS:
            if agg not in agg_data:
                continue

            runs = agg_data[agg]
            epochs_to_target = []
            for r in runs:
                acc = r['accuracy']
                reached = np.where(acc >= target_acc)[0]
                if len(reached) > 0:
                    epochs_to_target.append(reached[0])
                else:
                    epochs_to_target.append(len(acc))  # Never reached

            if epochs_to_target:
                data.append({
                    'attack': BYZ_LABELS.get(byz_type, byz_type),
                    'aggregation': AGG_LABELS[agg],
                    'agg_key': agg,
                    'mean_epochs': np.mean(epochs_to_target),
                    'std_epochs': np.std(epochs_to_target)
                })

    if not data:
        return

    df = pd.DataFrame(data)
    attacks = df['attack'].unique()
    x = np.arange(len(attacks))
    width = 0.15

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, agg in enumerate(AGGREGATIONS):
        agg_label = AGG_LABELS[agg]
        subset = df[df['aggregation'] == agg_label]

        means = []
        for attack in attacks:
            row = subset[subset['attack'] == attack]
            means.append(row['mean_epochs'].values[0] if len(row) > 0 else 0)

        ax.bar(x + i * width, means, width, label=agg_label, color=COLORS[agg])

    ax.set_xlabel('Attack Type', fontsize=12)
    ax.set_ylabel(f'Epochs to {target_acc:.0%} Accuracy', fontsize=12)
    ax.set_title(f'Convergence Speed: {model.upper()} on {dataset.upper()}', fontsize=14)
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(attacks, rotation=45, ha='right')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    save_path = os.path.join(output_dir, f'{model}_{dataset}_convergence_speed.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize experiment results')
    parser.add_argument('--out_dir', type=str, default='out', help='Results directory')
    parser.add_argument('--plot_dir', type=str, default='plots', help='Output directory for plots')
    parser.add_argument('--dataset', type=str, default=None, help='Specific dataset (e.g., mnist)')
    parser.add_argument('--model', type=str, default=None, help='Specific model (e.g., mlr)')
    parser.add_argument('--task', type=str, default='all',
                       choices=['all', 'curves', 'grid', 'bars', 'heatmap', 'ranking', 'convergence'],
                       help='Which visualization to generate')
    args = parser.parse_args()

    print("=" * 70)
    print("LOADING EXPERIMENT RESULTS")
    print("=" * 70)

    results = load_all_results(args.out_dir)

    if not results:
        print("No results found!")
        return

    # Determine datasets and models
    all_keys = list(results.keys())
    datasets = sorted(set(k[0] for k in all_keys))
    models = sorted(set(k[1] for k in all_keys))

    if args.dataset:
        datasets = [d for d in datasets if d.lower() == args.dataset.lower()]
    if args.model:
        models = [m for m in models if m.lower() == args.model.lower()]

    print(f"Datasets: {datasets}")
    print(f"Models: {models}")
    print(f"Total experiment configurations: {len(results)}")

    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    for dataset in datasets:
        for model in models:
            print(f"\nProcessing: {model}/{dataset}")

            if args.task in ['all', 'curves']:
                plot_training_curves_per_attack(results, dataset, model, args.plot_dir)

            if args.task in ['all', 'grid']:
                plot_combined_attack_grid(results, dataset, model, args.plot_dir)

            if args.task in ['all', 'bars']:
                plot_final_accuracy_comparison(results, dataset, model, args.plot_dir)

            if args.task in ['all', 'heatmap']:
                plot_heatmap(results, dataset, model, args.plot_dir)

            if args.task in ['all', 'convergence']:
                plot_convergence_speed(results, dataset, model, args.plot_dir)

    if args.task in ['all', 'ranking']:
        plot_aggregation_ranking(results, args.plot_dir)

    print("\n" + "=" * 70)
    print("VISUALIZATION COMPLETE")
    print(f"Plots saved to: {args.plot_dir}/")
    print("=" * 70)


if __name__ == '__main__':
    main()
