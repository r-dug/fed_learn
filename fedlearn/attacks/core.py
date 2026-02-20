"""Core Byzantine attacks: no-op, gaussian noise, gradient scaling."""

import tensorflow as tf


def no_byz(epoch, v, f, lr, perturbation):
    """No Byzantine attack - return gradients unchanged."""
    return v


def gaussian(epoch, v, f, lr, perturbation):
    """Gaussian attack - replace Byzantine gradients with random noise."""
    if f == 0:
        return v
    for i in range(f):
        v[i] = tf.random.normal(v[i].shape, mean=0.0, stddev=200.0)
    return v


def scale(epoch, v, f, lr, perturbation):
    """Scale attack - Byzantine workers scale their gradients."""
    if f == 0:
        return v
    scaling_factor = len(v)
    for i in range(f):
        v[i] = v[i] * scaling_factor
    return v
