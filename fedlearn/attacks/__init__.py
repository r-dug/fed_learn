"""Byzantine attack implementations and registry."""

from fedlearn.attacks.core import no_byz, gaussian, scale
from fedlearn.attacks.optimization import (
    trim_attack, krum_attack, MinMax, MinSum, lie, inner_product_manipulation,
)
from fedlearn.attacks.backdoor import model_replacement, model_replacement_adaptive

ATTACK_REGISTRY = {
    'none': no_byz,
    'label': no_byz,  # label flip is data-level, not gradient-level
    'gauss': gaussian,
    'trimAtt': trim_attack,
    'krumAtt': krum_attack,
    'scale': scale,
    'MinMax': MinMax,
    'MinSum': MinSum,
    'lie': lie,
    'modelReplace': model_replacement,
    'modelReplaceAdapt': model_replacement_adaptive,
    'IPM': inner_product_manipulation,
}


def get_attack(name):
    """Look up an attack function by name."""
    if name not in ATTACK_REGISTRY:
        raise ValueError(f"Unknown attack: '{name}'. Supported: {list(ATTACK_REGISTRY)}")
    return ATTACK_REGISTRY[name]
