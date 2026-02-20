"""Model definitions for federated learning experiments."""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers



def load_model(model_name, input_shape, num_classes=10):
    """Load and return a Keras model.

    Args:
        model_name: One of 'cnn', 'mlr', 'resnet'.
        input_shape: Tuple like (28, 28, 1).
        num_classes: Number of output classes.
    """
    if model_name == 'cnn':
        model = keras.Sequential([
            layers.Conv2D(30, kernel_size=3, activation='relu', input_shape=input_shape),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Conv2D(50, kernel_size=3, activation='relu'),
            layers.MaxPooling2D(pool_size=2, strides=2),
            layers.Flatten(),
            layers.Dense(100, activation='relu'),
            layers.Dense(num_classes)
        ])
    elif model_name == 'mlr':
        model = keras.Sequential([
            layers.Dense(64, activation='relu', input_shape=input_shape),
            layers.Dense(64, activation='relu'),
            layers.Flatten(trainable=False),  # Flatten non-trainable to avoid affecting gradients
            layers.Dense(num_classes)
        ])
    elif model_name == 'resnet':
        inputs = keras.Input(shape=input_shape)
        x = layers.Conv2D(16, 3, padding='same', activation='relu')(inputs)
        x = layers.BatchNormalization()(x)

        for filters in [16, 32, 64]:
            shortcut = x
            x = layers.Conv2D(filters, 3, padding='same', activation='relu')(x)
            x = layers.BatchNormalization()(x)
            x = layers.Conv2D(filters, 3, padding='same')(x)
            x = layers.BatchNormalization()(x)
            if shortcut.shape[-1] != filters:
                shortcut = layers.Conv2D(filters, 1, padding='same')(shortcut)
            x = layers.Add()([x, shortcut])
            x = layers.Activation('relu')(x)
            x = layers.MaxPooling2D(2)(x)

        x = layers.GlobalAveragePooling2D()(x)
        outputs = layers.Dense(num_classes)(x)
        model = keras.Model(inputs, outputs)
    else:
        raise ValueError(f"Unknown model: '{model_name}'. Supported: cnn, mlr, resnet")

    # Build model
    dummy_input = tf.zeros((1,) + input_shape)
    _ = model(dummy_input)

    return model
