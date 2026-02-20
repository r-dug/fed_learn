"""Model evaluation utilities."""

import tensorflow as tf

from fedlearn.constants import TRIGGER_PIXELS


def evaluate_accuracy(data, labels, model, dataset_name, trigger=False, target=None):
    """Evaluate model accuracy on given data.

    Args:
        data: Input tensor.
        labels: Label tensor.
        model: Keras model.
        dataset_name: Dataset name (for trigger pixel lookup).
        trigger: If True, apply trigger pattern and measure ASR.
        target: Target class for backdoor evaluation.
    """
    if trigger and dataset_name in TRIGGER_PIXELS:
        data = data.numpy().copy()
        labels = labels.numpy().copy()
        pixels = TRIGGER_PIXELS[dataset_name]
        num_channels = data.shape[-1]
        remaining_idx = []

        for i in range(len(data)):
            for r, c in pixels:
                if num_channels == 1:
                    data[i, r, c, 0] = 1.0
                else:
                    data[i, r, c, :] = 1.0
            if labels[i] != target:
                labels[i] = target
                remaining_idx.append(i)

        if remaining_idx:
            data = tf.constant(data[remaining_idx])
            labels = tf.constant(labels[remaining_idx])
        else:
            return 0.0

    predictions = model(data, training=False)
    pred_labels = tf.argmax(predictions, axis=1)
    accuracy = tf.reduce_mean(
        tf.cast(tf.equal(pred_labels, tf.cast(labels, tf.int64)), tf.float32)
    )
    return accuracy.numpy()
