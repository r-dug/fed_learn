import os
import sys
import time
import argparse
import hashlib
import threading
import subprocess
import concurrent.futures as cf
from tqdm import tqdm


# Parse command line arguments
parser = argparse.ArgumentParser(description='Run federated learning experiments')
parser.add_argument('--quick', action='store_true', help='Quick test mode (50 epochs, 1 seed)')
parser.add_argument('--threads', type=int, default=0, help='Number of parallel threads (0=auto based on GPU)')
parser.add_argument('--gpu', type=int, default=0, help='GPU index (-1 for CPU)')
parser.add_argument('--dry-run', action='store_true', help='Print configuration without running')
parser.add_argument('--target-util', type=float, default=0.85, help='Target GPU utilization (0.0-1.0, default 0.5)')
parser.add_argument('--mem-per-exp', type=int, default=0, help='Estimated MB per experiment (0=auto-detect)')
parser.add_argument('--verbose', action='store_true', help='Show individual experiment output (use with --threads=1)')
parser.add_argument('--resume', action='store_true', help='Skip experiments with existing non-empty output files')
parser.add_argument('--oom-retries', type=int, default=1, help='Retry count after CUDA OOM (default 1)')
run_args = parser.parse_args()


LOG_DIR = 'out/logs'
os.makedirs(LOG_DIR, exist_ok=True)

# Concurrency state for dynamic backoff after OOM failures.
SCHED_COND = threading.Condition()
LAUNCH_LOCK = threading.Lock()
DYNAMIC_LIMIT = 1
ACTIVE_EXPERIMENTS = 0
LAUNCH_PAUSE_SEC = 0.75
WAIT_POLL_SEC = 3.0


OOM_PATTERNS = (
    'cuda_error_out_of_memory',
    'cudaerrormemoryallocation',
    'resourceexhaustederror',
    'out_of_memory',
    'out of memory',
    'std::bad_alloc',
)


def get_gpu_memory_info(gpu_num=0):
    """Get GPU memory info (free, used, total) in MB using nvidia-smi."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.free,memory.used,memory.total', '--format=csv,nounits,noheader'],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            if gpu_num < len(lines):
                free, used, total = map(int, lines[gpu_num].split(','))
                return {'free': free, 'used': used, 'total': total}
    except Exception:
        pass
    return None


def mem_check(gpu_num=0):
    """Check GPU memory availability using nvidia-smi."""
    info = get_gpu_memory_info(gpu_num)
    if info:
        return info['free'] / info['total']
    return 1.0  # Assume available if we can't check


def estimate_memory_per_experiment(gpu_num=0, sample_command=None):
    """
    Estimate memory usage per experiment by running a short test.
    Returns estimated MB per experiment.
    """
    if sample_command is None:
        return None

    baseline = get_gpu_memory_info(gpu_num)
    if baseline is None:
        return None

    baseline_used = baseline['used']

    # Run a short version of the experiment to measure memory footprint.
    parts = sample_command.split()
    test_cmd = ' '.join('--epochs=2' if p.startswith('--epochs=') else p for p in parts)
    proc = subprocess.Popen(test_cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    max_used = baseline_used
    for _ in range(10):
        time.sleep(2)
        info = get_gpu_memory_info(gpu_num)
        if info and info['used'] > max_used:
            max_used = info['used']
        if proc.poll() is not None:
            break

    # Avoid long probe runs; memory sample is already collected.
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    else:
        proc.wait()

    estimated_mem = max_used - baseline_used
    return int(estimated_mem * 1.2) if estimated_mem > 0 else None


def calculate_optimal_threads(gpu_num=0, target_utilization=0.5, mem_per_exp_mb=None):
    """Calculate thread count to target a GPU memory utilization budget."""
    info = get_gpu_memory_info(gpu_num)
    if info is None:
        print('Warning: Could not query GPU memory, defaulting to 2 threads')
        return 2

    total_mb = info['total']
    current_used_mb = info['used']

    target_memory_mb = total_mb * target_utilization
    available_budget_mb = target_memory_mb - current_used_mb

    if available_budget_mb <= 0:
        print(
            f"Warning: GPU already at {current_used_mb/total_mb*100:.1f}% usage, "
            f"above target {target_utilization*100:.0f}%"
        )
        return 1

    if mem_per_exp_mb is None or mem_per_exp_mb <= 0:
        mem_per_exp_mb = 500
        print(f'Warning: Using fallback estimate of {mem_per_exp_mb}MB per experiment (auto-detect unavailable)')

    optimal_threads = max(1, int(available_budget_mb / mem_per_exp_mb))

    print(f"GPU Memory: {total_mb}MB total, {current_used_mb}MB used, {info['free']}MB free")
    print(f"Target utilization: {target_utilization*100:.0f}% ({target_memory_mb:.0f}MB)")
    print(f"Memory budget for experiments: {available_budget_mb:.0f}MB")
    print(f"Estimated {mem_per_exp_mb}MB per experiment -> {optimal_threads} parallel threads")

    return optimal_threads


def make_result_path(experiment):
    """Return output result file path for one experiment command."""
    para = (
        f"{experiment['seed']}+{experiment['dataset']}+{experiment['model']}+"
        f"bias{experiment['bias']}+epoch{experiment['epochs']}+local{experiment['local_round']}+"
        f"lr{experiment['lr']}+batch{experiment['batchsize']}+nwork{experiment['nworkers']}+"
        f"nbyz{experiment['nbyz']}+{experiment['byz_type']}+{experiment['aggregation']}+"
        f"{experiment['perturbation']}.txt"
    )
    out_dir = os.path.join('out', f"default+{experiment['dataset']}+byz_type_{experiment['byz_type']}")
    return os.path.join(out_dir, para)


def make_stderr_path(experiment, attempt_idx=0):
    """Build a unique stderr path per experiment, including retry suffix."""
    stem = (
        f"{experiment['dataset']}_{experiment['seed']}_{experiment['model']}_"
        f"{experiment['byz_type']}_{experiment['aggregation']}"
    )
    short_hash = hashlib.sha1(experiment['command'].encode('utf-8')).hexdigest()[:8]
    retry_suffix = '' if attempt_idx == 0 else f'_retry{attempt_idx}'
    return os.path.join(LOG_DIR, f'{stem}_{short_hash}{retry_suffix}.stderr')


def detect_oom(stderr_path):
    """Best-effort OOM detector from stderr logs."""
    if stderr_path is None or not os.path.isfile(stderr_path):
        return False

    try:
        file_size = os.path.getsize(stderr_path)
        with open(stderr_path, 'rb') as f:
            if file_size > 200_000:
                f.seek(-200_000, os.SEEK_END)
            content = f.read().decode('utf-8', errors='ignore').lower()
    except OSError:
        return False

    return any(pattern in content for pattern in OOM_PATTERNS)


def reduce_dynamic_limit():
    """Reduce dynamic concurrency by one, down to 1."""
    global DYNAMIC_LIMIT
    with SCHED_COND:
        old_limit = DYNAMIC_LIMIT
        if DYNAMIC_LIMIT > 1:
            DYNAMIC_LIMIT -= 1
        new_limit = DYNAMIC_LIMIT
        SCHED_COND.notify_all()
    return old_limit, new_limit


def select_probe_experiments(experiments):
    """Select representative experiments for memory probing."""
    probes = []

    def add_first(predicate):
        for exp in experiments:
            if predicate(exp):
                probes.append(exp)
                return

    add_first(lambda e: True)
    add_first(lambda e: e['aggregation'] == 'newMedian')
    add_first(lambda e: e['aggregation'] == 'krum')
    add_first(lambda e: e['model'] == 'resnet')

    deduped = []
    seen = set()
    for exp in probes:
        if exp is None:
            continue
        key = exp['command']
        if key in seen:
            continue
        deduped.append(exp)
        seen.add(key)

    return deduped


def run_if_mem(experiment, min_free_pct=0.25, verbose=False, gpu_num=0, oom_retries=1):
    """Run one experiment when launch conditions are safe."""
    global ACTIVE_EXPERIMENTS

    command = experiment['command']
    attempts = 0
    max_attempts = max(1, oom_retries + 1)
    last_returncode = 1
    last_stderr_path = None
    saw_oom = False

    while attempts < max_attempts:
        attempt_idx = attempts
        attempts += 1

        proc = None
        stderr_file = None
        stderr_path = None
        launched = False

        while not launched:
            with SCHED_COND:
                if ACTIVE_EXPERIMENTS >= DYNAMIC_LIMIT:
                    SCHED_COND.wait(timeout=WAIT_POLL_SEC)
                    continue

            with LAUNCH_LOCK:
                with SCHED_COND:
                    if ACTIVE_EXPERIMENTS >= DYNAMIC_LIMIT:
                        continue

                if mem_check(gpu_num=gpu_num) < min_free_pct:
                    pass
                else:
                    if verbose:
                        proc = subprocess.Popen(command, shell=True)
                    else:
                        stderr_path = make_stderr_path(experiment, attempt_idx)
                        stderr_file = open(stderr_path, 'w')
                        proc = subprocess.Popen(
                            command,
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=stderr_file,
                        )

                    with SCHED_COND:
                        ACTIVE_EXPERIMENTS += 1
                    launched = True
                    time.sleep(LAUNCH_PAUSE_SEC)

            if not launched:
                time.sleep(WAIT_POLL_SEC)

        try:
            last_returncode = proc.wait()
        finally:
            if stderr_file is not None:
                stderr_file.close()

            with SCHED_COND:
                ACTIVE_EXPERIMENTS = max(0, ACTIVE_EXPERIMENTS - 1)
                SCHED_COND.notify_all()

        if not verbose and stderr_path is not None:
            if os.path.isfile(stderr_path) and os.path.getsize(stderr_path) == 0:
                os.remove(stderr_path)
                stderr_path = None

        last_stderr_path = stderr_path

        if last_returncode == 0:
            break

        is_oom = detect_oom(last_stderr_path)
        if not is_oom or attempts >= max_attempts:
            break

        saw_oom = True
        old_limit, new_limit = reduce_dynamic_limit()
        if new_limit < old_limit:
            tqdm.write(f'OOM detected; reducing dynamic concurrency {old_limit} -> {new_limit}')
        else:
            tqdm.write('OOM detected; concurrency already at minimum (1)')

        time.sleep(10)

    return {
        'returncode': last_returncode,
        'stderr_path': last_stderr_path,
        'attempts': attempts,
        'oom_detected': saw_oom,
    }


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

if run_args.quick:
    print('*** QUICK TEST MODE ***')
    seed = [0]
    epochs = [1]
else:
    seed = [0, 1, 2, 3, 4]
    epochs = [2500]

dataset = ['cifar10', 'mnist', 'Fashion']
model = ['resnet', 'cnn', 'mlr']
bias = [0.5]
local_round = [10]
lr = [0.01]
batchsize = [32]

nworkers = [100]
nbyz = [2]

byz_type = ['none', 'gauss', 'label', 'trimAtt', 'krumAtt', 'scale', 'MinMax', 'MinSum', 'lie', 'modelReplace', 'modelReplaceAdapt', 'IPM']
aggregation = ['newMedian'] #'mean', 'trim', 'median', 'krum', 

perturbation = ['sgn']
pretrain_epochs = 10

gpu = [run_args.gpu]

# =============================================================================

all_experiments = []
for each_seed in seed:
    for each_dataset in dataset:
        for each_model in model:
            for each_bias in bias:
                for each_local_round in local_round:
                    for each_lr in lr:
                        for each_batchsize in batchsize:
                            for each_nworkers in nworkers:
                                for each_nbyz in nbyz:
                                    for each_byz_type in byz_type:
                                        for each_aggregation in aggregation:
                                            for each_perturbation in perturbation:
                                                command = (
                                                    f"{sys.executable} train.py"
                                                    + f" --dataset={each_dataset}"
                                                    + f" --model={each_model}"
                                                    + f" --bias={each_bias}"
                                                    + f" --seed={each_seed}"
                                                    + f" --epochs={epochs[0]}"
                                                    + f" --local_round={each_local_round}"
                                                    + f" --lr={each_lr}"
                                                    + f" --batchsize={each_batchsize}"
                                                    + f" --nworkers={each_nworkers}"
                                                    + f" --nbyz={each_nbyz}"
                                                    + f" --byz_type={each_byz_type}"
                                                    + f" --aggregation={each_aggregation}"
                                                    + f" --perturbation={each_perturbation}"
                                                    + f" --pretrain_epochs={pretrain_epochs}"
                                                    + f" --gpu={gpu[0]}"
                                                )
                                                if not run_args.verbose:
                                                    command += ' --quiet'

                                                exp = {
                                                    'command': command,
                                                    'seed': each_seed,
                                                    'dataset': each_dataset,
                                                    'model': each_model,
                                                    'bias': each_bias,
                                                    'epochs': epochs[0],
                                                    'local_round': each_local_round,
                                                    'lr': each_lr,
                                                    'batchsize': each_batchsize,
                                                    'nworkers': each_nworkers,
                                                    'nbyz': each_nbyz,
                                                    'byz_type': each_byz_type,
                                                    'aggregation': each_aggregation,
                                                    'perturbation': each_perturbation,
                                                }
                                                exp['result_path'] = make_result_path(exp)
                                                all_experiments.append(exp)

def estimate_experiment_log_size(epochs:int)->int:
    """create a temp log file and measure its size to estimate the log size for a given number of epochs"""
    import tempfile
    import random

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        for epoch in range(epochs):
            acc = random.uniform(0, 1)
            loss = random.uniform(0, 2)
            log_line = f"{acc:.4f} {loss:.4f}\n"
            tmp.write(log_line.encode('utf-8'))
        tmp.flush()
        size = os.path.getsize(tmp.name)
    os.remove(tmp.name)
    return size    

experiments = all_experiments
skipped_existing = 0
if run_args.resume:
    pending = []
    for exp in experiments:
        if os.path.isfile(exp['result_path']) and os.path.getsize(exp['result_path']) > 0:
            skipped_existing += 1
        else:
            pending.append(exp)
    experiments = pending

# =============================================================================
# DETERMINE THREAD COUNT (deferred — needs experiment list for auto-detect probe)
# =============================================================================

if run_args.threads > 0:
    threads = run_args.threads
    print(f'Using specified thread count: {threads}')
elif run_args.gpu == -1:
    import multiprocessing
    threads = max(1, multiprocessing.cpu_count() // 2)
    print(f'CPU mode: using {threads} threads')
else:
    mem_estimate = run_args.mem_per_exp if run_args.mem_per_exp > 0 else None

    if mem_estimate is None and experiments:
        probe_exps = select_probe_experiments(experiments)
        print('Probing GPU memory usage with representative experiments...')
        probe_estimates = []
        for idx, probe_exp in enumerate(probe_exps, 1):
            label = f"{probe_exp['dataset']}/{probe_exp['model']}/{probe_exp['aggregation']}"
            est = estimate_memory_per_experiment(
                gpu_num=run_args.gpu,
                sample_command=probe_exp['command'],
            )
            if est:
                probe_estimates.append(est)
                print(f'  Probe {idx}/{len(probe_exps)} ({label}): ~{est}MB')
            else:
                print(f'  Probe {idx}/{len(probe_exps)} ({label}): failed')

        if probe_estimates:
            mem_estimate = max(probe_estimates)
            print(f'Using worst-case estimate: ~{mem_estimate}MB per experiment')
        else:
            print('All probes failed, falling back to default estimate')

    threads = calculate_optimal_threads(
        gpu_num=run_args.gpu,
        target_utilization=run_args.target_util,
        mem_per_exp_mb=mem_estimate,
    )

threads = max(1, threads)
with SCHED_COND:
    DYNAMIC_LIMIT = threads

# Print experiment summary
total_experiments = len(experiments)
print('=' * 70)
print('FEDERATED LEARNING EXPERIMENT SUITE')
print('=' * 70)
print(f'  Total configured: {len(all_experiments)} ({len(seed)} seeds × {len(dataset)} datasets × {len(byz_type)} attacks × {len(aggregation)} aggs)')
if run_args.resume:
    print(f'  Resume skipped: {skipped_existing}')
print(f'  Pending: {total_experiments}')
print(f'  Epochs: {epochs[0]} | Workers: {nworkers[0]} ({nbyz[0]} Byzantine)')
print(f'  Parallel: {threads} threads | GPU util target: {run_args.target_util*100:.0f}%')
print()
print('=' * 70)
sys.stdout.flush()

# Dry run mode - just print config and exit
if run_args.dry_run:
    print('\n*** DRY RUN MODE - Not executing experiments ***')
    print('\nSample commands:')
    for i, exp in enumerate(experiments[:3]):
        print(f"  [{i+1}] {exp['command']}")
    if len(experiments) > 3:
        print(f'  ... and {len(experiments) - 3} more')
    raise SystemExit(0)

if total_experiments == 0:
    print('No pending experiments to run.')
    raise SystemExit(0)

# Pre-download datasets to avoid race conditions between parallel processes
needed_datasets = set(exp['dataset'] for exp in experiments)
if needed_datasets:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    print('Pre-downloading datasets...')
    from fedlearn.data import _load_dataset_safe
    for ds_name in sorted(needed_datasets):
        try:
            _load_dataset_safe(ds_name)
            print(f'  {ds_name}: OK')
        except Exception as e:
            print(f'  {ds_name}: download failed ({e}), experiments will retry')
    print()

# Track progress
completed = 0
failed = 0
start_time = time.time()

# Calculate minimum free memory threshold (inverse of target utilization)
min_free_threshold = 1.0 - run_args.target_util

# Overall progress bar
pbar = tqdm(
    total=total_experiments,
    desc='Overall',
    position=0,
    ncols=80,
    bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]',
)


cwd = os.getcwd()
failure_log_file = os.path.join(cwd, 'failed_runs.txt')
if not os.path.isfile(failure_log_file):
    with open(failure_log_file, 'w'):
        pass

# Run experiments with thread pool
with cf.ThreadPoolExecutor(max_workers=threads) as executor:
    futures = {
        executor.submit(
            run_if_mem,
            exp,
            min_free_threshold,
            run_args.verbose,
            run_args.gpu,
            run_args.oom_retries,
        ): exp
        for exp in experiments
    }

    for future in cf.as_completed(futures):
        exp = futures.pop(future)
        command = exp['command']

        try:
            result = future.result()
            completed += 1
            if result['returncode'] != 0:
                failed += 1
                with open(failure_log_file, 'a') as f:
                    f.write(
                        f"Failed: returncode={result['returncode']} attempts={result['attempts']} "
                        f"oom_detected={result['oom_detected']}\n"
                    )
                    f.write(f"Command: {command}\n")
                    if result['stderr_path']:
                        f.write(f"Stderr: {result['stderr_path']}\n")
                    f.write('\n')
        except Exception as exc:
            failed += 1
            completed += 1
            tqdm.write(f'FAILED: {exc}')
            with open(failure_log_file, 'a') as f:
                f.write(f'Failed: exception={exc}\n')
                f.write(f'Command: {command}\n\n')

        with SCHED_COND:
            current_limit = DYNAMIC_LIMIT
        pbar.update(1)
        pbar.set_postfix({'failed': failed, 'limit': current_limit})

pbar.close()

# Final summary
total_time = time.time() - start_time
print('\n' + '=' * 70)
print('COMPLETE')
print(f'  Time: {total_time/3600:.2f} hours | Success: {completed - failed}/{total_experiments} | Failed: {failed}')
print('=' * 70)
