"""Training loop and GPU/seed setup for federated learning experiments."""

import os
import random
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tqdm import tqdm

from fedlearn.constants import BACKDOOR_ATTACKS, INPUT_SHAPES, BACKDOOR_TARGET
from fedlearn.config import result_dir, result_path, byzantine_log_path
from fedlearn.models import load_model
from fedlearn.data import (
    load_and_prepare_dataset, assign_data_to_workers,
    inject_backdoor_triggers, apply_label_flip,
)
from fedlearn.evaluate import evaluate_accuracy
from fedlearn.attacks import get_attack
from fedlearn.aggregation import get_aggregation

logger = logging.getLogger(__name__)


def setup_gpu(gpu_index):
    """Configure GPU visibility and memory growth.

    Returns:
        Device string like '/GPU:0' or '/CPU:0'.
    """
    if gpu_index == -1:
        tf.config.set_visible_devices([], 'GPU')
        return '/CPU:0'

    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            tf.config.set_visible_devices(gpus[gpu_index], 'GPU')
            tf.config.experimental.set_memory_growth(gpus[gpu_index], True)
            return f'/GPU:{gpu_index}'
        except RuntimeError as e:
            logger.warning(f"GPU setup failed: {e}")
    return '/CPU:0'


def set_seeds(seed):
    """Set random seeds for reproducibility."""
    tf.random.set_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def warmup(model, worker_data, worker_labels, batch_size, loss_func, optimizer,
           local_round=1):
    """Run warmup training on the first worker's data.

    BUG FIX #5: The original code had an unconditional `break` making
    local_round have no effect. Now properly loops local_round times.
    """
    for local_epoch in range(local_round):
        if len(worker_data[0]) == 0:
            continue
        indices = np.random.choice(
            len(worker_data[0]),
            size=min(batch_size, len(worker_data[0])),
            replace=False,
        )
        data = tf.constant(worker_data[0][indices])
        label = tf.constant(worker_labels[0][indices])

        with tf.GradientTape() as tape:
            outputs = model(data, training=True)
            loss = loss_func(label, outputs)

        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))


def _pretrain_centralized(model, x_train, y_train, epochs, batch_size,
                          loss_func, optimizer, quiet=False):
    """Pretrain model as a central server using the full training set.

    Runs standard mini-batch SGD over x_train/y_train for the given number
    of epochs with no Byzantine corruption.  Provides a meaningful model
    initialization so that FL-round gradient norms reflect task signal rather
    than random-init noise.
    """
    n = len(x_train)
    pbar = tqdm(range(epochs), desc='Pretraining', ncols=80, disable=quiet)
    for _ in pbar:
        indices = np.random.permutation(n)
        for start in range(0, n, batch_size):
            batch_idx = indices[start:start + batch_size]
            data = tf.gather(x_train, batch_idx)
            labels = tf.gather(y_train, batch_idx)
            with tf.GradientTape() as tape:
                outputs = model(data, training=True)
                loss = loss_func(labels, outputs)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
    pbar.close()


def train(cfg):
    """Main federated learning training loop.

    Args:
        cfg: ExperimentConfig instance.
    """
    # Setup
    device = setup_gpu(cfg.gpu)
    set_seeds(cfg.seed)

    if not cfg.quiet:
        logger.info(f"Using device: {device}")

    input_shape = INPUT_SHAPES[cfg.dataset]
    model = load_model(cfg.model, input_shape, cfg.classes)

    lr = cfg.lr
    optimizer = keras.optimizers.SGD(learning_rate=lr, momentum=cfg.momentum)
    loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    # Ensure output directories exist
    out_dir = result_dir(cfg)
    os.makedirs(out_dir, exist_ok=True)

    # Determine train size
    train_size = 5000 if cfg.dataset == 'cifar10' else 6000
    test_size = 500

    # Load data
    x_train, y_train, x_test, y_test = load_and_prepare_dataset(
        cfg.dataset, train_size=train_size, test_size=test_size, seed=cfg.seed,
    )

    # Assign data to workers
    each_worker_data, each_worker_label = assign_data_to_workers(
        x_train, y_train, cfg.nworkers, cfg.classes, cfg.bias, cfg.seed,
    )

    # Model initialisation: explicit checkpoint > cached pretrain > legacy warmup
    if cfg.checkpoint:
        if not os.path.exists(cfg.checkpoint):
            raise FileNotFoundError(f"Checkpoint not found: {cfg.checkpoint}")
        model.load_weights(cfg.checkpoint)
        if not cfg.quiet:
            print(f'Loaded checkpoint from {cfg.checkpoint}')
    elif cfg.pretrain_epochs > 0:
        ckpt_dir = os.path.join('out', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        pretrain_ckpt = os.path.join(
            ckpt_dir,
            f'{cfg.dataset}_{cfg.model}_seed{cfg.seed}_pretrain{cfg.pretrain_epochs}.weights.h5',
        )
        if os.path.exists(pretrain_ckpt):
            model.load_weights(pretrain_ckpt)
            if not cfg.quiet:
                print(f'Loaded pretrain checkpoint from {pretrain_ckpt}')
        else:
            if not cfg.quiet:
                print(f'Pretraining for {cfg.pretrain_epochs} epochs (centralized SGD)...')
            _pretrain_centralized(
                model, x_train, y_train,
                cfg.pretrain_epochs, cfg.batchsize,
                loss_func, optimizer, cfg.quiet,
            )
            model.save_weights(pretrain_ckpt)
            if not cfg.quiet:
                print(f'Saved pretrain checkpoint to {pretrain_ckpt}')
    else:
        if not cfg.quiet:
            print('Warming up...')
        warmup(model, each_worker_data, each_worker_label,
               cfg.batchsize, loss_func, optimizer, cfg.local_round)

    # Inject backdoor triggers for backdoor attacks
    if cfg.byz_type in BACKDOOR_ATTACKS:
        inject_backdoor_triggers(
            each_worker_data, each_worker_label, cfg.nbyz, cfg.dataset,
        )

    # Label flipping attack
    if cfg.byz_type == 'label':
        apply_label_flip(each_worker_label, cfg.nbyz)

    # Resolve attack and aggregation functions via registries
    attack_fn = get_attack(cfg.byz_type)
    agg_fn = get_aggregation(cfg.aggregation)
    perturbation = cfg.perturbation
    byz_log = byzantine_log_path(cfg) if cfg.aggregation == 'newMedian' else None

    # Progress bar
    exp_id = f"{cfg.dataset[:3]}|{cfg.byz_type[:6]:6s}|{cfg.aggregation[:6]:6s}|s{cfg.seed}"
    test_acc_list = []

    # BUG FIX #4: Use result_path() for both periodic and final saves
    save_path = result_path(cfg)

    pbar = tqdm(
        range(cfg.epochs),
        desc=f"{exp_id} acc=-.----",
        position=cfg.progress_pos,
        leave=True,
        ncols=80,
        disable=cfg.quiet,
        bar_format='{desc} |{bar}| {n_fmt}/{total_fmt}',
    )

    for each_epoch in pbar:
        byz = attack_fn if each_epoch > 0 else get_attack('none')

        grad_list = []
        for each_worker in range(cfg.nworkers):
            if len(each_worker_data[each_worker]) == 0:
                continue

            available_samples = each_worker_data[each_worker].shape[0]
            sample_size = min(cfg.batchsize, available_samples)
            use_replace = sample_size < cfg.batchsize
            minibatch = np.random.choice(available_samples, size=sample_size, replace=use_replace)

            data = tf.constant(each_worker_data[each_worker][minibatch])
            label = tf.constant(each_worker_label[each_worker][minibatch])

            with tf.GradientTape() as tape:
                outputs = model(data, training=True)
                loss = loss_func(label, outputs)

            gradients = tape.gradient(loss, model.trainable_variables)
            valid_grads = [g for g in gradients if g is not None]
            if len(valid_grads) == 0:
                continue
            flat_grad = tf.concat(
                [tf.reshape(tf.cast(g, tf.float32), [-1]) for g in valid_grads], axis=0,
            )
            grad_list.append(flat_grad)

        if len(grad_list) == 0:
            continue

        grad_list = [tf.reshape(g, (-1, 1)) for g in grad_list]
        # Aggregation — newMedian takes extra byz_log_file kwarg
        if cfg.aggregation == 'newMedian':
            agg_gradient = agg_fn(
                each_epoch, grad_list, model, lr / cfg.batchsize,
                perturbation, cfg.nbyz, byz,
                byz_log_file=byz_log,
            )
        else:
            agg_gradient = agg_fn(
                each_epoch, grad_list, model, lr / cfg.batchsize,
                perturbation, cfg.nbyz, byz,
            )

        # Update model parameters
        idx = 0
        for var in model.trainable_variables:
            var_size = tf.size(var).numpy()
            update = tf.reshape(agg_gradient[idx:idx + var_size], var.shape)
            var.assign_sub(lr * update)
            idx += var_size

        del grad_list

        # Validation
        if each_epoch % cfg.interval == 0 or each_epoch == cfg.epochs - 1:
            if cfg.byz_type in BACKDOOR_ATTACKS:
                test_accuracy = evaluate_accuracy(x_test, y_test, model, cfg.dataset)
                attack_success_rate = evaluate_accuracy(
                    x_test, y_test, model, cfg.dataset,
                    trigger=True, target=BACKDOOR_TARGET,
                )
                test_acc_list.append((test_accuracy, attack_success_rate))
                pbar.set_description(
                    f"{exp_id} acc={test_accuracy:.4f} ASR={attack_success_rate:.4f}",
                )
            else:
                test_accuracy = evaluate_accuracy(x_test, y_test, model, cfg.dataset)
                test_acc_list.append(test_accuracy)
                pbar.set_description(f"{exp_id} acc={test_accuracy:.4f}")

            # Periodic save to correct path (BUG FIX #4)
            np.savetxt(save_path, test_acc_list, fmt='%.4f')

    pbar.close()

    # Auto-save checkpoint for base case
    if cfg.byz_type == 'none' and cfg.seed == 0 and cfg.aggregation == 'mean':
        ckpt_dir = os.path.join('out', 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, f'{cfg.dataset}_{cfg.model}.weights.h5')
        model.save_weights(ckpt_path)
        if not cfg.quiet:
            print(f'Saved base case checkpoint to {ckpt_path}')

    # Final save (periodic save already keeps file up to date; this ensures
    # the last epoch is captured even if it wasn't an interval boundary)
    np.savetxt(save_path, test_acc_list, fmt='%.4f')
