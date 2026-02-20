#!/usr/bin/env python3
"""Thin CLI entry point for federated learning experiments.

Replaces main_OUR.py with a clean argparse → ExperimentConfig → train() pipeline.
"""

import os
import argparse
import logging

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

from fedlearn.config import ExperimentConfig
from fedlearn.training import train


def parse_args():
    parser = argparse.ArgumentParser(description='Federated learning experiment')
    parser.add_argument("--dataset", type=str, default='mnist')
    parser.add_argument("--classes", type=int, default=10)
    parser.add_argument("--batchsize", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--bias", type=float, default=0.5)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--model", type=str, default='cnn')
    parser.add_argument("--momentum", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=733)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--log", type=str, default='log.txt')
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--local_round", type=int, default=1)
    parser.add_argument("--nworkers", type=int, default=10)
    parser.add_argument("--nbyz", type=int, default=2)
    parser.add_argument("--byz_type", type=str, default='none')
    parser.add_argument("--aggregation", type=str, default='mean')
    parser.add_argument("--perturbation", type=str, default='sgn')
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--quiet", action='store_true')
    parser.add_argument("--progress_pos", type=int, default=0)
    parser.add_argument("--pretrain_epochs", type=int, default=0,
                        help="Epochs of centralized SGD pretraining before FL (checkpoint cached)")
    return parser.parse_args()


def main():
    args = parse_args()

    # Set up logging
    filehandler = logging.FileHandler(args.log)
    streamhandler = logging.StreamHandler()
    root_logger = logging.getLogger('')
    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(filehandler)
    root_logger.addHandler(streamhandler)

    cfg = ExperimentConfig.from_args(args)
    train(cfg)


if __name__ == '__main__':
    main()
