"""Basic aggregation rules: mean, trimmed mean, median, Krum."""

import tensorflow as tf

from fedlearn.attacks._krum_helpers import score
from fedlearn.attacks.core import no_byz


def mean(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz):
    """Average aggregation rule."""
    param_list = gradients
    param_list = byz(epoch, param_list, f, lr, perturbation)

    stacked = tf.concat(param_list, axis=1)
    mean_result = tf.reduce_mean(stacked, axis=-1)
    return mean_result


def trim(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz):
    """Coordinate-wise trimmed mean aggregation rule."""
    param_list = gradients
    param_list = byz(epoch, param_list, f, lr, perturbation)

    stacked = tf.concat(param_list, axis=1)
    sorted_array = tf.sort(stacked, axis=-1)

    n = len(param_list)
    b = f
    m = n - b * 2

    trimmed = sorted_array[:, b:(b + m)]
    trim_mean = tf.reduce_mean(trimmed, axis=-1)
    return trim_mean


def median(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz):
    """Coordinate-wise median aggregation rule."""
    param_list = gradients
    param_list = byz(epoch, param_list, f, lr, perturbation)

    stacked = tf.concat(param_list, axis=1)
    sorted_array = tf.sort(stacked, axis=-1)

    n = sorted_array.shape[-1]
    if n % 2 == 1:
        median_result = sorted_array[:, n // 2]
    else:
        median_result = (sorted_array[:, n // 2 - 1] + sorted_array[:, n // 2]) / 2

    return median_result


def krum(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz):
    """Krum aggregation rule."""
    param_list = gradients
    num_params = len(param_list)

    q = f
    if num_params - f - 2 <= 0:
        q = num_params - 3

    param_list = byz(epoch, param_list, f, lr, perturbation)

    v = tf.concat(param_list, axis=1)
    scores = tf.constant([score(gradient, v, q) for gradient in param_list])
    min_idx = int(tf.argmin(scores).numpy())
    krum_result = tf.reshape(param_list[min_idx], shape=(-1,))

    return krum_result
