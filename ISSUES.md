# Issues Found During Codebase Inspection

## Critical - Will Cause Crashes or Silent Failures

### 1. `activate.sh` points to wrong virtual environment path
**File:** `activate.sh:8`
- **Problem:** `VENV_DIR="/home/richard/Documents/UofL/.venv"` but `setup_env.sh` creates the venv at `$SCRIPT_DIR/.venv` which resolves to `/home/richard/Documents/UofL/tensorflow/.venv`.
- **Effect:** `source activate.sh` will fail to activate the environment (or activate a stale/wrong one).

### 2. Undefined `agg_gradient` for unrecognized aggregation method
**File:** `main_OUR.py:447-456`
- **Problem:** The if/elif chain for `args.aggregation` has no `else` clause. If an unrecognized aggregation string is passed, `agg_gradient` is never assigned, causing a `NameError` at line 462.
- **Effect:** Crash with no useful error message.

### 3. `load_model` returns `None` for unrecognized model names
**File:** `main_OUR.py:149-189`
- **Problem:** If `model_name` is not `'cnn'`, `'mlr'`, or `'resnet'`, the function implicitly returns `None`. Line 205 (`model(dummy_input)`) will then crash with a confusing `TypeError`.

### 4. `input_shape` undefined for unsupported datasets
**File:** `main_OUR.py:193-199`
- **Problem:** If `dataset` is not `'mnist'`, `'Fashion'`, or `'cifar10'`, `input_shape` is never defined, causing `NameError` at line 201.

### 5. Periodic save to `out/` assumes directory exists
**File:** `main_OUR.py:485`
- **Problem:** `np.savetxt('out/' + paraString, ...)` writes directly to `out/` on every validation interval. While the final save (lines 494-495) creates subdirectories with `os.makedirs`, this mid-training save does not. On a fresh clone or a cluster node, `out/` may not exist.
- **Effect:** Crash on first validation checkpoint.

## Correctness - Will Produce Wrong or Misleading Results

### 6. Test set size of 256 is too small for reliable accuracy estimates
**File:** `main_OUR.py:224`
- **Problem:** `test_size = 256` uses only 256 of the available 10,000 test samples. With 10 classes, that's ~25 samples per class on average.
- **Effect:** Accuracy estimates will have high variance across runs, making it harder to draw statistically significant conclusions, especially for 2550 epochs of training.

### 7. Backdoor result parsing only handles `scale` attack
**File:** `aggregate_results.py:81`, `tabulate.py:132-148`
- **Problem:** `main_OUR.py` produces 2-column output (accuracy + ASR) for all backdoor attacks: `scale`, `modelReplace`, and `modelReplaceAdapt` (line 326: `BACKDOOR_ATTACKS = ['scale', 'modelReplace', 'modelReplaceAdapt']`). But `aggregate_results.py` only checks `byz_type == 'scale'` for 2-column handling. The other backdoor attack results will be mis-parsed.
- **Effect:** `modelReplace` and `modelReplaceAdapt` results will be read incorrectly (interpreted as a 1D array when they are 2D), potentially crashing or reporting wrong numbers.

### 8. Learning rate inconsistency between aggregation and model update
**File:** `main_OUR.py:448-456, 464`
- **Problem:** The aggregation functions receive `lr / args.batchsize` as their `lr` parameter. This value is then forwarded to Byzantine attack functions, which use it for internal scaling (e.g., `krum_attack` uses `stop_threshold = 0.00001 * 2 / lr`). However, the actual model update at line 464 uses `lr` (without the batchsize division). The attacks are calibrated to a different learning rate than what is actually applied.
- **Effect:** Byzantine attacks like `krum_attack` may not behave as intended, potentially weakening or strengthening attacks in ways that skew results.

### 9. `b_log.txt` concurrent write race condition
**File:** `tf_aggregation.py:150-155`
- **Problem:** The `newMedian` function appends to `b_log.txt` on every epoch. When `run.py` launches parallel experiments, all processes will append to the same file simultaneously with no file locking.
- **Effect:** Corrupted/interleaved log entries. The log becomes unusable for analyzing Byzantine detection accuracy.

## Environment / Deployment

### 10. `run.py` suppresses all stderr output from experiments
**File:** `run.py:146-152`
- **Problem:** In non-verbose mode (the default for parallel runs), both `stdout` and `stderr` are redirected to `DEVNULL`. If an experiment crashes (OOM, import error, CUDA failure), the only signal is a non-zero return code with no diagnostic information.
- **Recommendation:** Redirect stderr to a per-experiment log file instead of discarding it entirely.

### 11. `setup_env.sh` has interactive prompts - incompatible with batch jobs
**File:** `setup_env.sh:210, 220, 241`
- **Problem:** The script uses `read -p` for interactive confirmation prompts ("Install CUDA toolkit? [y/N]", "Continue with CPU-only setup? [Y/n]", "Recreate virtual environment? [y/N]"). These will hang indefinitely in a non-interactive batch job on a compute cluster.

### 12. Subprocess `python3` may not resolve to venv Python
**File:** `run.py:217`
- **Problem:** Experiments are launched as `python3 main_OUR.py ...` subprocesses. On a cluster, if the environment is not activated before running `run.py`, `python3` may resolve to the system Python, which won't have the required packages installed.
- **Recommendation:** Use `sys.executable` instead of hardcoding `python3` to ensure the subprocess uses the same Python interpreter as the parent process.

## Minor

### 13. `estimate_b` is imported but never used
**File:** `main_OUR.py:15`
- `from b_estimation import estimate_b` is imported but never called anywhere. The `b_estimation.py` module is a placeholder that always returns 0.

### 14. `num_inputs` is assigned but never used
**File:** `main_OUR.py:99`
- `num_inputs = 28 * 28` is set conditionally but never referenced.

### 15. `mem_check()` in `main_OUR.py` is a non-functional placeholder
**File:** `main_OUR.py:53-63`
- The function always returns either `1.0` (no GPU) or `0.5` (GPU present), never actually checking memory. The `run.py` script has a real implementation using `nvidia-smi`, but `main_OUR.py` does not use it. The "waiting for GPU" loop at lines 67-70 is effectively dead code.
