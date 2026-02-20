"""Backdoor attacks: model replacement and adaptive model replacement."""

import tensorflow as tf


def model_replacement(epoch: int, v: list, f: int, lr: float, perturbation: str)-> list:
    """Model Replacement Backdoor Attack (Bagdasaryan et al., 2020).

    Scales Byzantine gradients to ensure the backdoor is learned despite averaging.
    """
    if f == 0:
        return v

    vi_shape = v[0].shape
    n = len(v)

    v_tran = tf.concat(v, axis=1)

    byz_grads = v_tran[:, :f]
    g_byz = tf.reduce_mean(byz_grads, axis=1, keepdims=True)

    honest_grads = v_tran[:, f:]
    g_honest = tf.reduce_mean(honest_grads, axis=1, keepdims=True)

    gamma = float(n) / float(f)
    g_mal = gamma * g_byz - (gamma - 1) * g_honest

    honest_norm = tf.norm(g_honest)
    mal_norm = tf.norm(g_mal)
    max_norm_ratio = 10.0

    if mal_norm > max_norm_ratio * honest_norm:
        g_mal = g_mal * (max_norm_ratio * honest_norm / mal_norm)

    for i in range(f):
        noise = tf.random.normal(vi_shape, mean=0.0, stddev=0.001)
        noise = noise * tf.norm(g_mal) * 0.01
        v[i] = g_mal + noise

    return v


def model_replacement_adaptive(epoch: int, v: list, f: int, lr: float, perturbation: str)-> list:
    """Adaptive Model Replacement Backdoor Attack with evasion techniques.

    Gradually increases attack strength and projects gradients to stay
    within statistical bounds of honest gradients.
    """
    if f == 0:
        return v

    vi_shape = v[0].shape
    n = len(v)

    v_tran = tf.concat(v, axis=1)

    byz_grads = v_tran[:, :f]
    g_byz = tf.reduce_mean(byz_grads, axis=1, keepdims=True)

    honest_grads = v_tran[:, f:]
    g_honest_mean = tf.reduce_mean(honest_grads, axis=1, keepdims=True)
    g_honest_std = tf.math.reduce_std(honest_grads, axis=1, keepdims=True) + 1e-8

    warmup_epochs = 30
    if epoch < warmup_epochs:
        attack_strength = 0.3 + 0.7 * (epoch / warmup_epochs)
    else:
        attack_strength = 1.0

    gamma = float(n) / float(f) * attack_strength
    g_mal_raw = gamma * g_byz - (gamma - 1) * g_honest_mean

    k_std = 4.0
    upper_bound = g_honest_mean + k_std * g_honest_std
    lower_bound = g_honest_mean - k_std * g_honest_std

    g_mal_projected = tf.clip_by_value(g_mal_raw, lower_bound, upper_bound)

    for i in range(f):
        diversity_scale = 1.0 + 0.05 * (i - f / 2) / max(f, 1)
        noise = tf.random.normal(vi_shape, mean=0.0, stddev=1.0)
        noise = noise * g_honest_std * 0.05

        v[i] = g_mal_projected * diversity_scale + noise

    return v
