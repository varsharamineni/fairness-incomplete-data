import numpy as np

def compute_disparate_impact_from_joint(joint, classifier_probs):
    """
    Computes disparate impact given a joint distribution with string keys 'psoe'
    and a classifier with p(y=1 | s, o) probabilities.

    Args:
        joint (dict): Joint distribution with keys like 'p000' representing p(s, o, e)
        classifier_probs (dict): Dictionary with keys (s, o) and values p(y=1 | s, o)

    Returns:
        dict: Contains p(y=1|e=0), p(y=1|e=1), and the disparate impact ratio
    """
    # Convert joint keys to (s, o, e)
    psoe = {tuple(map(int, k[1:])): float(v) for k, v in joint.items()}

    # Compute p(e)
    p_e = {e: sum(p for (s, o, e_), p in psoe.items() if e_ == e) for e in [0, 1]}

    # Compute p(s, o | e)
    p_so_given_e = {
        (s, o, e): p / p_e[e]
        for (s, o, e), p in psoe.items()
        if p_e[e] > 0
    }

    # Compute p(y=1 | e)
    p_y1_given_e = {
        e: sum(
            classifier_probs[(s, o)] * p_so_given_e.get((s, o, e), 0.0)
            for (s, o) in classifier_probs
        )
        for e in [0, 1]
    }

    disparate_impact = p_y1_given_e[0] / p_y1_given_e[1] if p_y1_given_e[1] > 0 else np.nan

    return disparate_impact


def compute_demographic_disparity_from_joint(joint, classifier_probs):
    """
    Computes demographic disparity: P(ŷ=1|e=0) - P(ŷ=1|e=0)

    Args:
        joint (dict): keys like 'p001' representing p(s, o, e)
        classifier_probs (dict): {(s, o): P(ŷ=1 | s, o)}

    Returns:
        float: Demographic disparity
    """
    # Convert joint keys to (s, o, e)
    psoe = {tuple(map(int, k[1:])): float(v) for k, v in joint.items()}

    # Compute p(e)
    p_e = {e: sum(p for (s, o, e_), p in psoe.items() if e_ == e) for e in [0, 1]}

    # Compute p(s, o | e)
    p_so_given_e = {
        (s, o, e): p / p_e[e]
        for (s, o, e), p in psoe.items()
        if p_e[e] > 0
    }

    # Compute p(y=1 | e)
    p_y1_given_e = {
        e: sum(
            classifier_probs[(s, o)] * p_so_given_e.get((s, o, e), 0.0)
            for (s, o) in classifier_probs
        )
        for e in [0, 1]
    }

    dd = p_y1_given_e[0] - p_y1_given_e[1]

    return dd