import os
import warnings
import argparse
import logging
import numpy as np
import pandas as pd

import visualize_results as viz

from fedlearn.constants import BACKDOOR_ATTACKS, ALL_BYZ_TYPES, ALL_AGGREGATIONS

# Set up logging for error tracking
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def check_files():
    """Collect all result files from output directory."""
    files = []
    for root, dir, f in os.walk(OUT):
        if not root == OUT:
            files += f
    return list(set(files))


def create_table(model: str, dataset: str):
    """
    Create a table with aggregation method as column, attack as row, and error as cell value.
    Backdoor attacks are split into separate '(top 1)' and '(backdoor)' rows.
    """
    byz_types = list(ALL_BYZ_TYPES)
    aggregations = list(ALL_AGGREGATIONS)

    loss_dict = {}

    for aggregation in aggregations:
        agg_losses = []
        for byz in byz_types:
            result_files = [f for f in FILES if (aggregation + "+" in f and model + "+" in f and dataset + "+" in f and byz + "+" in f)]
            if byz in BACKDOOR_ATTACKS:
                total_loss = [0, 0]
            else:
                total_loss = 0
            count = 0
            for r in result_files:
                try:
                    with open(OUT + f"default+{dataset}+byz_type_{byz}/" + r) as f:
                        lines = f.readlines()
                        if not lines:
                            continue
                        line = lines[-1]
                        if byz in BACKDOOR_ATTACKS:
                            losses = [1 - float(val) for val in line.split()]
                            total_loss[0] += sum(losses[0::2])
                            total_loss[1] += sum(losses[1::2])
                            count += len(losses[0::2])
                        else:
                            losses = [1 - float(val) for val in line.split()]
                            total_loss += sum(losses)
                            count += len(losses)
                except Exception as e:
                    logger.warning(f"Error processing {r}: {e}")
                    continue

            this_loss = None
            if count > 0:
                with warnings.catch_warnings():
                    warnings.filterwarnings("error", category=RuntimeWarning)
                    try:
                        this_loss = np.array(total_loss) / count
                    except RuntimeWarning:
                        this_loss = None

            if byz in BACKDOOR_ATTACKS:
                if this_loss is not None:
                    agg_losses += list(this_loss)
                else:
                    agg_losses += [None, None]
            else:
                agg_losses.append(this_loss)

        loss_dict[aggregation] = agg_losses

    idx = ['none', 'gauss', 'label', 'trimAtt', 'krumAtt',
           'scale (top 1)', 'scale (backdoor)',
           'MinMax', 'MinSum', 'lie',
           'modelReplace (top 1)', 'modelReplace (backdoor)',
           'modelReplaceAdapt (top 1)', 'modelReplaceAdapt (backdoor)',
           'IPM']
    try:
        return pd.DataFrame(loss_dict, index=idx)
    except Exception as e:
        logger.error(f"Error creating DataFrame: {e}")
        print(e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", help="which task to perform", type=str, default='tabulate')
    parser.add_argument("--out_dir", type=str, default='out', help='Results directory')
    parser.add_argument("--plot_dir", type=str, default='plots', help='Output directory for plots')
    args = parser.parse_args()

    OUT = os.getcwd() + f'/{args.out_dir}/'
    FILES = check_files()

    models = ['cnn', 'mlr', 'resnet']
    datasets = ['mnist', 'Fashion', 'cifar10']

    def valid_combo(model, dataset):
        if dataset == "cifar10" and model != 'resnet':
            return False
        if dataset != "cifar10" and model == 'resnet':
            return False
        return True

    with warnings.catch_warnings():
        if args.task == 'tabulate':
            for model in models:
                for dataset in datasets:
                    if not valid_combo(model, dataset):
                        continue
                    df = create_table(model=model, dataset=dataset)
                    try:
                        file_path = f"{OUT}{model}_{dataset}.xlsx"
                        df.to_excel(file_path)
                        print(f"**** {model}  || {dataset} : Excel written ****")
                    except Exception as e:
                        logger.error(f"Error writing Excel for {model}/{dataset}: {e}")
                        print(e)

        if args.task == 'graph_error':
            results = viz.load_all_results(args.out_dir)
            for model in models:
                for dataset in datasets:
                    if not valid_combo(model, dataset):
                        continue
                    viz.plot_training_curves_per_attack(results, dataset, model, args.plot_dir)

        if args.task == 'plot_summary':
            results = viz.load_all_results(args.out_dir)
            for model in models:
                for dataset in datasets:
                    if not valid_combo(model, dataset):
                        continue
                    viz.plot_final_accuracy_comparison(results, dataset, model, args.plot_dir)

        if args.task == 'plot_average_summary':
            results = viz.load_all_results(args.out_dir)
            viz.plot_aggregation_ranking(results, args.plot_dir)
