"""Optimization-based Byzantine attacks."""

import tensorflow as tf
from scipy.stats import norm

from fedlearn.attacks._krum_helpers import krum_helper


def trim_attack(epoch, v, f, lr, perturbation):
    """Trim attack - Byzantine workers send extreme values to skew trimmed mean."""
    if f == 0:
        return v
    vi_shape = v[0].shape
    v_tran = tf.concat(v, axis=1)
    maximum_dim = tf.reshape(tf.reduce_max(v_tran, axis=1), vi_shape)
    minimum_dim = tf.reshape(tf.reduce_min(v_tran, axis=1), vi_shape)
    direction = tf.sign(tf.reduce_sum(v_tran, axis=-1, keepdims=True))
    directed_dim = (tf.cast(direction > 0, tf.float32) * minimum_dim +
                    tf.cast(direction < 0, tf.float32) * maximum_dim)

    for i in range(f):
        random_12 = 1 + tf.random.uniform(shape=vi_shape)
        v[i] = directed_dim * (
            tf.cast(direction * directed_dim > 0, tf.float32) / random_12 +
            tf.cast(direction * directed_dim < 0, tf.float32) * random_12
        )
    return v


def krum_attack(epoch, v, f, lr, perturbation):
    """Krum attack - craft Byzantine gradients to be selected by Krum."""
    if f == 0:
        return v
    epsilon = 0.01
    vi_shape = v[0].shape
    _, original_dir = krum_helper(v, f)
    original_dir = tf.reshape(original_dir, vi_shape)

    lambda_ = 0.25
    for i in range(f):
        v[i] = -lambda_ * tf.sign(original_dir)

    min_idx, _ = krum_helper(v, f)
    stop_threshold = 0.00001 * 2 / lr

    while min_idx >= f and lambda_ > stop_threshold:
        lambda_ = lambda_ / 2
        for i in range(f):
            v[i] = -lambda_ * tf.sign(original_dir)
        min_idx, _ = krum_helper(v, f)

    v[0] = -lambda_ * tf.sign(original_dir)
    for i in range(1, f):
        random_raw = tf.random.uniform(shape=vi_shape) - 0.5
        random_norm = tf.random.uniform(()).numpy() * epsilon / lr
        randomness = random_raw * random_norm / tf.norm(random_raw)
        v[i] = -lambda_ * tf.sign(original_dir) + randomness
    return v


def MinMax(epoch, v, f, lr, perturbation):
    """MinMax attack - minimize maximum distance to benign gradients."""
    if f == 0:
        return v
    lambda_ = 100.0
    threshold_diff = 1e-5
    lambda_fail = lambda_
    lambda_succ = 0.0

    v_tran = tf.concat(v, axis=1)

    if perturbation == 'sgn':
        deviation = tf.sign(tf.reduce_sum(v_tran, axis=-1, keepdims=True))
    elif perturbation == 'uv':
        deviation1 = tf.reduce_mean(v_tran, axis=-1, keepdims=True)
        deviation = deviation1 / tf.norm(deviation1)
    elif perturbation == 'std':
        n = len(v)
        e_mu = tf.reduce_mean(v_tran, axis=1)
        e_sigma = tf.sqrt(tf.reduce_sum(tf.square(v_tran - tf.reshape(e_mu, (-1, 1))), axis=1) / n)
        deviation = tf.reshape(e_sigma, v[0].shape)
    else:
        deviation = tf.sign(tf.reduce_sum(v_tran, axis=-1, keepdims=True))

    model_re = tf.reduce_mean(v_tran, axis=-1, keepdims=True)
    max_distance = 0.0

    for grad in v:
        distance = tf.reduce_max(tf.norm(v_tran - grad, axis=0) ** 2)
        max_distance = max(max_distance, distance.numpy())

    iter_count = 0
    while abs(lambda_succ - lambda_) > threshold_diff and iter_count < 5:
        iter_count += 1
        mal_update = model_re - lambda_ * deviation
        distance = tf.norm(v_tran - tf.reshape(mal_update, v[0].shape), axis=0) ** 2
        max_d = tf.reduce_max(distance)

        if max_d <= max_distance:
            lambda_succ = lambda_
            lambda_ = lambda_ + lambda_fail / 2
        else:
            lambda_ = lambda_ - lambda_fail / 2

        lambda_fail = lambda_fail / 2

    mal_update = model_re - lambda_succ * deviation
    for i in range(f):
        v[i] = mal_update

    return v


def MinSum(epoch, v, f, lr, perturbation):
    """MinSum attack - minimize sum of distances to benign gradients."""
    if f == 0:
        return v
    lambda_ = 100.0
    threshold_diff = 1e-5
    lambda_fail = lambda_
    lambda_succ = 0.0

    v_tran = tf.concat(v, axis=1)

    if perturbation == 'sgn':
        deviation = tf.sign(tf.reduce_sum(v_tran, axis=-1, keepdims=True))
    elif perturbation == 'uv':
        deviation1 = tf.reduce_mean(v_tran, axis=-1, keepdims=True)
        deviation = deviation1 / tf.norm(deviation1)
    elif perturbation == 'std':
        n = len(v)
        e_mu = tf.reduce_mean(v_tran, axis=1)
        e_sigma = tf.sqrt(tf.reduce_sum(tf.square(v_tran - tf.reshape(e_mu, (-1, 1))), axis=1) / n)
        deviation = tf.reshape(e_sigma, v[0].shape)
    else:
        deviation = tf.sign(tf.reduce_sum(v_tran, axis=-1, keepdims=True))

    model_re = tf.reduce_mean(v_tran, axis=-1, keepdims=True)

    max_distance = 0.0
    for grad in v:
        distance = tf.reduce_sum(tf.norm(v_tran - grad, axis=0) ** 2)
        max_distance = max(max_distance, distance.numpy())

    iter_count = 0
    while abs(lambda_succ - lambda_) > threshold_diff and iter_count < 5:
        iter_count += 1
        mal_update = model_re - lambda_ * deviation
        distance = tf.norm(v_tran - tf.reshape(mal_update, v[0].shape), axis=0) ** 2
        sum_distance = tf.reduce_sum(distance)

        if sum_distance <= max_distance:
            lambda_succ = lambda_
            lambda_ = lambda_ + lambda_fail / 2
        else:
            lambda_ = lambda_ - lambda_fail / 2

        lambda_fail = lambda_fail / 2

    mal_update = model_re - lambda_succ * deviation
    for i in range(f):
        v[i] = mal_update

    return v


def lie(epoch, v, f, lr, perturbation):
    """LIE (Little Is Enough) attack."""
    if f == 0:
        return v
    v_tran = tf.concat(v, axis=1)
    all_mean = tf.reduce_mean(v_tran, axis=-1, keepdims=True)

    v_np = v_tran.numpy()
    all_std = v_np.std(axis=1)
    all_std = tf.constant(all_std, dtype=tf.float32)
    all_std = tf.reshape(all_std, all_mean.shape)

    number_worker = len(v)
    s = int(number_worker / 2 + 1) - f
    a = (number_worker - f - s) / (number_worker - f)
    z = norm.ppf(a)

    for i in range(f):
        v[i] = all_mean - z * all_std

    return v


def inner_product_manipulation(epoch, v, f, lr, perturbation):
    """Inner Product Manipulation (IPM) Attack."""
    if f == 0:
        return v

    vi_shape = v[0].shape
    v_tran = tf.concat(v, axis=1)

    honest_grads = v_tran[:, f:]
    g_honest = tf.reduce_mean(honest_grads, axis=1, keepdims=True)
    g_honest_norm = tf.norm(g_honest)

    honest_norms = tf.norm(honest_grads, axis=0)
    target_norm = tf.reduce_mean(honest_norms)

    g_mal_direction = -g_honest / (g_honest_norm + 1e-8)
    g_mal_base = g_mal_direction * target_norm

    random_component = tf.random.normal(vi_shape)
    proj = tf.reduce_sum(random_component * g_honest) / (g_honest_norm ** 2 + 1e-8)
    orthogonal = random_component - proj * g_honest
    orthogonal = orthogonal / (tf.norm(orthogonal) + 1e-8)

    epsilon = 0.3
    g_mal = g_mal_base * (1 - epsilon) + orthogonal * target_norm * epsilon

    for i in range(f):
        noise = tf.random.normal(vi_shape, stddev=0.01) * target_norm
        v[i] = g_mal + noise

    return v
