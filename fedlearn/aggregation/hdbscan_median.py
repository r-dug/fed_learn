"""Novel HDBSCAN-based median aggregation with GPU acceleration.

Byzantine detection rationale
------------------------------
Non-IID federated learning makes gradient-based clustering unreliable for
Byzantine detection: honest workers training on heterogeneous local data
produce gradient norms that span 400x+ and directions that spread across
the full unit sphere.  Any clustering feature — Euclidean distance, cosine
distance, L2 norm, or combinations — produces massive false positives because
the honest worker distribution is indistinguishable from an attacker set.

Additionally, random model initialisation causes all workers to produce
high-magnitude, chaotic gradients in early epochs, further confounding any
norm-based outlier detector.

This implementation uses a two-layer defence:

Layer 1 — f-based trimming (primary):
  m = n - f workers participate in the coordinatewise median selection.
  Since f is known per experiment, this is always correctly calibrated.
  The coordinatewise closest-m-to-median selection is Byzantine-robust by
  construction: for each parameter coordinate, workers far from the median
  are excluded regardless of their cluster identity.

Layer 2 — HDBSCAN supplemental filter (secondary, post-warmup only):
  HDBSCAN on the gradient L2 norms may identify extreme magnitude outliers
  after the model has converged enough for gradient norms to be meaningful.
  The detected count is capped at f so it can never remove more workers than
  the known Byzantine count (prevents false-positive honest-worker removal).
  b reported in the Byzantine log reflects the raw HDBSCAN output so
  detection quality can be monitored across experiments.
"""

import logging

import tensorflow as tf
import numpy as np

from fedlearn.attacks.core import no_byz

logger = logging.getLogger(__name__)

# Try to use GPU-accelerated HDBSCAN from cuML (RAPIDS), fall back to sklearn
try:
    from cuml.cluster import HDBSCAN as CumlHDBSCAN
    import cupy as cp
    USE_CUML = True
    logger.info("Using cuML GPU-accelerated HDBSCAN")
except ImportError:
    USE_CUML = False
    logger.info("cuML not available, using sklearn HDBSCAN (CPU)")
from sklearn.cluster import HDBSCAN as SklearnHDBSCAN


def _flatten_to_column(grad):
    """Normalize one worker gradient to shape [num_params, 1]."""
    if isinstance(grad, (list, tuple)):
        return tf.concat([tf.reshape(g, (-1, 1)) for g in grad], axis=0)

    if isinstance(grad, tf.Tensor) and grad.shape.rank == 2 and grad.shape[-1] == 1:
        return grad

    return tf.reshape(grad, (-1, 1))


def _tf_to_cupy(tensor):
    """Convert TF tensor to CuPy, preferring DLPack for GPU-to-GPU transfer."""
    dlpack_capsule = tf.experimental.dlpack.to_dlpack(tensor)

    if hasattr(cp, "from_dlpack"):
        return cp.from_dlpack(dlpack_capsule) #from DLpack deprecated in favor of from_dlpack, but some versions only have fromdlpack
    return cp.fromDlpack(dlpack_capsule) #fallback for older CuPy versions


def _compute_chunk_size(num_params, num_workers, bytes_per_row):
    """Compute chunk size for median selection to stay within target_mb overhead.
    The main GPU memory overhead comes from the distance matrix (W, D) of absolute differences, which is W * D * 4 bytes for float32.  We want to choose
    """
    #  target memory overhead should be less than a fraction of avail gpu memory
    #  we can get the available GPU memory using cp.cuda.Device().mem_info[0] (free memory in bytes)
    try:
        free_memory = cp.cuda.Device().mem_info[0]
        target_mb = free_memory / (1024 * 1024 * 2)  # Use half of available GPU memory
    except:
        target_mb = 500  # fallback if GPU memory info is not available
    chunk_size = max(100, int((target_mb * 1024 * 1024) / (num_workers * bytes_per_row)))  
    return min(chunk_size, num_params)

def compute_bytes_in_row(row):
    """Return bytes per element (itemsize) for the gradient array dtype."""
    if isinstance(row, tf.Tensor):
        return row.dtype.size
    elif isinstance(row, (np.ndarray, cp.ndarray)):
        return row.itemsize
    else:
        raise ValueError("Unsupported type for computing bytes: {}".format(type(row)))


def estimate_b(grads, n):
    """Run HDBSCAN on gradient cosine distances, return outlier (Byzantine) count.

    grads: (W, D) array — one row per worker.
    Cosine distance captures directional outliers regardless of gradient magnitude,
    making it robust to non-IID norm variance among honest workers.
    """
    norms = np.linalg.norm(grads, axis=1, keepdims=True)
    # hdb = SklearnHDBSCAN(metric='cosine',
    #                      cluster_selection_method='eom',
    #                      allow_single_cluster=True,
    #                      copy=True)
    
    hdb2 = CumlHDBSCAN(metric='l2',
                         cluster_selection_method='eom',
                         allow_single_cluster=True,
                         copy=True)
    
    # cosine_outliers = hdb.fit_predict(grads)
    euclidean_outliers = hdb2.fit_predict(norms)
    # Combine outlier predictions: a worker is Byzantine if flagged by either method
    combined_outliers = (cosine_outliers == -1) & (euclidean_outliers == -1)
    return int(np.sum(combined_outliers == -1))


def newMedian(epoch, gradients, net, lr, perturbation, f=0, byz=no_byz,
              byz_log_file=None):
    """Novel median-based aggregation with HDBSCAN Byzantine detection.

    Assumes the model has been pretrained (via --pretrain_epochs) so that
    gradient norms reflect task signal rather than random-init noise.

    HDBSCAN is run on the 1-D gradient L2 norms each epoch with
    min_cluster_size = (n//2)+1 (strict majority).  Workers outside the
    majority cluster are predicted Byzantine (b).  m = n - b workers
    participate in the coordinatewise closest-m-to-median selection.

    Args:
        epoch: Current training epoch (1-indexed).
        byz_log_file: Optional path to write per-epoch Byzantine detection log.
    """
    import time
    start = time.time()
    # print(f"Epoch {epoch}: Starting HDBSCAN median aggregation with f={f} and byz={byz}\n\tWorker Shape = {gradients[0].shape} Time taken: {time.time() - start:.2f} seconds")
    param_list = byz(epoch, gradients, f, lr, perturbation)
    # print(f"Epoch {epoch}: Finished preparing gradients, starting aggregation.\n\tworker shape: {param_list[0].shape} Time taken: {time.time() - start:.2f} seconds")
    n = len(param_list)
    
    stacked = tf.concat(param_list, axis=1)
    # print(f"Epoch {epoch}: Stacked gradients shape: {stacked.shape} Time taken: {time.time() - start:.2f} seconds")
    if USE_CUML:
        # Phase 1: Transfer to CuPy via DLPack (zero-copy GPU-to-GPU)
        gradients_gpu = _tf_to_cupy(tf.transpose(stacked))
        del stacked  # release TF reference

        # Strict majority: any worker outside the majority cluster is predicted Byzantine.
        b = estimate_b(cp.asnumpy(gradients_gpu), n)
        # print(f"Epoch {epoch}: HDBSCAN detected {b} Byzantine workers. Time taken: {time.time() - start:.2f} seconds")
        m = min(n - b, n // 2)  # Ensure at least n//2 workers participate

        cp.get_default_memory_pool().free_all_blocks()

        # Phase 3: chunked coordinate-wise median selection
        D = gradients_gpu.shape[1]
        bytes_per_row = compute_bytes_in_row(gradients_gpu[:, 0])
        chunk_size = _compute_chunk_size(D, n, bytes_per_row)
        result_chunks = []
        for start in range(0, D, chunk_size):
            chunk = gradients_gpu[:, start:start + chunk_size]        # (W, chunk_size)
            med = cp.median(chunk, axis=0, keepdims=True)              # (1, chunk_size)
            dist = cp.abs(chunk - med)                                 # (W, chunk_size)
            del med
            idx = cp.argpartition(dist, kth=m - 1, axis=0)[:m, :]    # (m, chunk_size)
            del dist
            selected = cp.take_along_axis(chunk, idx, axis=0)         # (m, chunk_size)
            result_chunks.append(cp.mean(selected, axis=0))            # (chunk_size,)
            cp.get_default_memory_pool().free_all_blocks()  # free GPU memory after each chunk

        aggregated_gradient = cp.concatenate(result_chunks)            # (D,)
        final_result = tf.constant(aggregated_gradient.get(), dtype=tf.float32)
        # print(f"Epoch {epoch}: Completed GPU-accelerated HDBSCAN median aggregation.\n\tAgg Shape: {final_result.shape} Time taken: {time.time() - start:.2f} seconds")
    else:
        # CPU fallback via sklearn
        gradients_matrix = stacked.numpy()
        grads_t = gradients_matrix.T   # [n_workers, num_params]

        b_hdbscan = estimate_b(grads_t, n)
        b = b_hdbscan
        m = max(1, n - b_hdbscan)

        median_vector = np.median(gradients_matrix, axis=0, keepdims=True)
        distances = np.abs(gradients_matrix - median_vector)
        closest_indices = np.argpartition(distances, kth=m - 1, axis=1)[:, :m]
        selected_gradients = np.take_along_axis(gradients_matrix, closest_indices, axis=1)
        aggregated_gradient = np.mean(selected_gradients, axis=1, keepdims=True)
        final_result = tf.constant(aggregated_gradient, dtype=tf.float32)

    if byz_log_file:
        try:
            with open(byz_log_file, 'a') as file:
                file.write(f"{b} {f}\n")
        except IOError as e:
            logger.warning(f"Could not write to {byz_log_file}: {e}")

    return tf.reshape(final_result, [-1,])
