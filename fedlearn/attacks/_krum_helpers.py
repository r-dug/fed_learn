"""Shared Krum scoring helpers used by both attacks and aggregation."""

import tensorflow as tf


def score(gradient, v, f):
    """Compute Krum score for a gradient."""
    num_neighbours = int(v.shape[1] - 2 - f)
    distances = tf.reduce_sum(tf.square(v - gradient), axis=0)
    sorted_distance = tf.sort(distances)
    return tf.reduce_sum(sorted_distance[1:(1 + num_neighbours)]).numpy()


def krum_helper(v, f):
    """Helper function for Krum selection."""
    if len(v[0].shape) > 1:
        v_tran = tf.concat(v, axis=1)
    else:
        v_tran = tf.stack(v, axis=1)

    scores = tf.constant([score(gradient, v_tran, f) for gradient in v])
    min_idx = int(tf.argmin(scores).numpy())
    krum_result = tf.reshape(v[min_idx], shape=(-1,))
    return min_idx, krum_result
