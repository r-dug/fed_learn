"""Single source of truth for all shared constants."""

# ── Attack & aggregation type lists ──────────────────────────────────────────

BACKDOOR_ATTACKS = frozenset(['scale', 'modelReplace', 'modelReplaceAdapt'])

ALL_BYZ_TYPES = [
    'none', 'gauss', 'label', 'trimAtt', 'krumAtt',
    'scale', 'MinMax', 'MinSum', 'lie',
    'modelReplace', 'modelReplaceAdapt', 'IPM',
]

ALL_AGGREGATIONS = ['mean', 'trim', 'median', 'krum', 'newMedian']

# ── Dataset constants ────────────────────────────────────────────────────────

SUPPORTED_DATASETS = ['mnist', 'Fashion', 'cifar10']

INPUT_SHAPES = {
    'mnist':   (28, 28, 1),
    'Fashion': (28, 28, 1),
    'cifar10': (32, 32, 3),
}

DEFAULT_TRAIN_SIZE = 50000
DEFAULT_TEST_SIZE = 10000

# ── Backdoor trigger pixels (row, col) per dataset ──────────────────────────

TRIGGER_PIXELS = {
    'mnist':   [(26, 26), (24, 26), (26, 24), (25, 25)],
    'Fashion': [(26, 26), (24, 26), (26, 24), (25, 25)],
    'cifar10': [(30, 30), (28, 30), (30, 28), (29, 29)],
}

BACKDOOR_TARGET = 0

# ── Output paths ─────────────────────────────────────────────────────────────

OUTPUT_DIR_PREFIX = 'out'

# ── Display constants for visualization ──────────────────────────────────────

AGGREGATION_DISPLAY = {
    'mean':      'Mean',
    'trim':      'Trimmed Mean',
    'median':    'Median',
    'krum':      'Krum',
    'newMedian': 'HDBSCAN Median (Ours)',
}

BYZ_TYPE_DISPLAY = {
    'none':              'No Attack',
    'gauss':             'Gaussian',
    'label':             'Label Flip',
    'trimAtt':           'Trim Attack',
    'krumAtt':           'Krum Attack',
    'scale':             'Scaling',
    'MinMax':            'MinMax',
    'MinSum':            'MinSum',
    'lie':               'LIE',
    'modelReplace':      'Model Replace',
    'modelReplaceAdapt': 'Adaptive Model Replace',
    'IPM':               'IPM',
}

AGGREGATION_COLORS = {
    'mean':      '#1f77b4',
    'trim':      '#ff7f0e',
    'median':    '#2ca02c',
    'krum':      '#d62728',
    'newMedian': '#9467bd',
}
