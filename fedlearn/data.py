"""Dataset loading, worker assignment, and trigger injection."""

import os
import fcntl
import shutil
import glob as _glob

import numpy as np
import tensorflow as tf
from tensorflow import keras

from fedlearn.constants import TRIGGER_PIXELS, BACKDOOR_TARGET

_DATASET_LOADERS = {
    'mnist': keras.datasets.mnist.load_data,
    'Fashion': keras.datasets.fashion_mnist.load_data,
    'cifar10': keras.datasets.cifar10.load_data,
}

_DATASET_CACHE_GLOBS = {
    'mnist': 'mnist*',
    'Fashion': 'fashion*',
    'cifar10': 'cifar*',
}


def _load_dataset_safe(name):
    """Load a Keras dataset with inter-process locking.

    Uses LOCK_SH for concurrent reads and upgrades to LOCK_EX only when
    the cache is missing or corrupt and a re-download is required.
    """
    loader = _DATASET_LOADERS[name]
    lock_path = os.path.join(os.path.expanduser('~'), '.keras', 'datasets', f'.{name}.lock')
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_fd = open(lock_path, 'w')

    try:
        fcntl.flock(lock_fd, fcntl.LOCK_SH)
        try:
            return loader()
        except Exception:
            pass

        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        try:
            return loader()
        except Exception:
            pass

        cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
        for p in _glob.glob(os.path.join(cache_dir, _DATASET_CACHE_GLOBS[name])):
            try:
                if os.path.isfile(p):
                    os.unlink(p)
                elif os.path.isdir(p):
                    shutil.rmtree(p)
            except OSError:
                pass

        return loader()
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


def load_and_prepare_dataset(name, train_size=6000, test_size=500, seed=733):
    """Load dataset, normalize, shuffle, and return tensors.

    Returns:
        (x_train, y_train, x_test, y_test) as tf.Tensors.
    """
    (x_train, y_train), (x_test, y_test) = _load_dataset_safe(name)

    if name == 'cifar10':
        y_train = y_train.flatten()
        y_test = y_test.flatten()

    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0

    if name in ['mnist', 'Fashion']:
        x_train = x_train[..., np.newaxis]
        x_test = x_test[..., np.newaxis]

    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    rng = np.random.RandomState(seed)
    train_indices = rng.permutation(len(x_train))[:train_size]
    test_indices = rng.permutation(len(x_test))[:test_size]
    x_train = x_train[train_indices]
    y_train = y_train[train_indices]
    x_test = x_test[test_indices]
    y_test = y_test[test_indices]

    return (
        tf.constant(x_train), tf.constant(y_train),
        tf.constant(x_test), tf.constant(y_test),
    )


def assign_data_to_workers(x_train, y_train, num_workers, num_classes=10,
                           bias=0.5, seed=733):
    """Assign training data to workers with label bias.

    Returns:
        (each_worker_data, each_worker_label) — lists of numpy arrays.
    """
    x_np = x_train.numpy() if isinstance(x_train, tf.Tensor) else x_train
    y_np = y_train.numpy() if isinstance(y_train, tf.Tensor) else y_train

    other_group_size = (1 - bias) / 9.0
    worker_per_group = num_workers / 10

    each_worker_data = [[] for _ in range(num_workers)]
    each_worker_label = [[] for _ in range(num_workers)]

    for i in range(len(x_np)):
        x = x_np[i]
        y_val = int(y_np[i])

        upper_bound = y_val * (1 - bias) / 9.0 + bias
        lower_bound = y_val * (1 - bias) / 9.0
        rd = np.random.random_sample()

        if rd > upper_bound:
            worker_group = int(np.floor((rd - upper_bound) / other_group_size) + y_val + 1)
        elif rd < lower_bound:
            worker_group = int(np.floor(rd / other_group_size))
        else:
            worker_group = y_val

        worker_group = worker_group % 10

        rd = np.random.random_sample()
        selected_worker = int(worker_group * worker_per_group + int(np.floor(rd * worker_per_group)))
        selected_worker = min(selected_worker, num_workers - 1)

        each_worker_data[selected_worker].append(x)
        each_worker_label[selected_worker].append(y_val)

    each_worker_data = [np.stack(d) if len(d) > 0 else np.empty((0, *d[0].shape)) for d in each_worker_data]
    each_worker_label = [np.stack(l) if len(l) > 0 else np.empty((0,), dtype=np.int32) for l in each_worker_label]

    # Shuffle worker order
    random_order = np.random.RandomState(seed=seed).permutation(num_workers)
    each_worker_data = [each_worker_data[i] for i in random_order]
    each_worker_label = [each_worker_label[i] for i in random_order]

    return each_worker_data, each_worker_label


def inject_backdoor_triggers(worker_data, worker_labels, nbyz, dataset_name,
                             target=BACKDOOR_TARGET):
    """Inject backdoor triggers into Byzantine workers' data.

    Duplicates each sample, adding trigger to every other copy and
    setting its label to the target class.
    """
    pixels = TRIGGER_PIXELS.get(dataset_name)
    if pixels is None:
        return

    num_channels = worker_data[0].shape[-1] if len(worker_data[0]) > 0 else 1

    for worker_id in range(nbyz):
        if len(worker_data[worker_id]) == 0:
            continue
        worker_data[worker_id] = np.repeat(worker_data[worker_id], repeats=2, axis=0)
        worker_labels[worker_id] = np.repeat(worker_labels[worker_id], repeats=2, axis=0)

        for example_id in range(0, len(worker_data[worker_id]), 2):
            for r, c in pixels:
                if num_channels == 1:
                    worker_data[worker_id][example_id, r, c, 0] = 1.0
                else:
                    worker_data[worker_id][example_id, r, c, :] = 1.0
            worker_labels[worker_id][example_id] = target


def apply_label_flip(worker_labels, nbyz):
    """Apply label flipping attack to Byzantine workers."""
    for i in range(nbyz):
        worker_labels[i] = 9 - worker_labels[i]
