"""Aggregation method implementations and registry."""

from fedlearn.aggregation.basic import mean, trim, median, krum
from fedlearn.aggregation.hdbscan_median import newMedian

AGGREGATION_REGISTRY = {
    'mean': mean,
    'trim': trim,
    'median': median,
    'krum': krum,
    'newMedian': newMedian,
}


def get_aggregation(name):
    """Look up an aggregation function by name."""
    if name not in AGGREGATION_REGISTRY:
        raise ValueError(f"Unknown aggregation: '{name}'. Supported: {list(AGGREGATION_REGISTRY)}")
    return AGGREGATION_REGISTRY[name]
