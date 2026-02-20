import os
import numpy as np
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from fedlearn.config import ExperimentConfig
from fedlearn.training import setup_gpu, set_seeds
from fedlearn.models import load_model
from fedlearn.data import load_and_prepare_dataset, assign_data_to_workers
from fedlearn.constants import INPUT_SHAPES
import tensorflow.keras as keras

cfg = ExperimentConfig(dataset='mnist', model='mlr', seed=42, nworkers=100, nbyz=1,
                       batchsize=32, lr=0.01, bias=0.5)
setup_gpu(0)
set_seeds(42)
model = load_model(cfg.model, INPUT_SHAPES[cfg.dataset], 10)
model.load_weights('out/checkpoints/mnist_mlr_seed42_pretrain100.weights.h5')

loss_func = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
x_train, y_train, _, _ = load_and_prepare_dataset('mnist', 6000, 500, 42)
each_worker_data, each_worker_label = assign_data_to_workers(
    x_train, y_train, 100, 10, 0.5, 42)

norms = []
for w in range(100):
    d = each_worker_data[w]
    idx = np.random.choice(len(d), size=min(32, len(d)), replace=False)
    data = tf.constant(d[idx])
    label = tf.constant(each_worker_label[w][idx])
    with tf.GradientTape() as tape:
        out = model(data, training=True)
        loss = loss_func(label, out)
    grads = tape.gradient(loss, model.trainable_variables)
    flat = tf.concat([tf.reshape(g, [-1]) for g in grads], 0)
    norms.append(float(tf.norm(flat).numpy()))

norms = np.array(norms)
print(f"min={norms.min():.3f}  max={norms.max():.3f}  ratio={norms.max()/norms.min():.1f}x")
print(f"cv={norms.std()/norms.mean():.3f}")
print(f"p5={np.percentile(norms,5):.3f}  p50={np.percentile(norms,50):.3f}  p95={np.percentile(norms,95):.3f}")
print(f"p5-p95 ratio: {np.percentile(norms,95)/np.percentile(norms,5):.1f}x")
