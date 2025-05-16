import numpy as np

def generate_random_joint():
    keys = ['p000', 'p001', 'p010', 'p011', 'p100', 'p101', 'p110', 'p111']
    probs = np.random.dirichlet(np.ones(8))
    return dict(zip(keys, probs))

def generate_random_classifier_probs(seed=None):
    rng = np.random.default_rng(seed)  # Creates a reproducible RNG if seed is set
    classifier_probs = {}
    for o in [0, 1]:
        alpha = np.array([1, 1])  # Uniform over [0, 1]
        probs = rng.dirichlet(alpha)
        classifier_probs[(0, o)] = probs[0]  # p(y = 1 | s=0, o)
        classifier_probs[(1, o)] = probs[1]  # p(s = 1 | s=1, o)
    return classifier_probs



# Generate joint distribution systematically (example with 3 variables: S, O, E)
def generate_systematic_joint():
    joint = {}
    for p_000 in np.arange(0.05, 1.0, 0.05):
        for p_001 in np.arange(0.05, 1.0 - p_000, 0.05):
            for p_010 in np.arange(0.05, 1.0 - p_000 - p_001, 0.05):
                p_011 = 1 - (p_000 + p_001 + p_010)
                if p_011 >= 0:
                    joint = {
                        'p000': p_000,
                        'p001': p_001,
                        'p010': p_010,
                        'p011': p_011,
                        # Assuming symmetric distribution for other values
                        'p100': p_000,
                        'p101': p_001,
                        'p110': p_010,
                        'p111': p_011,
                    }
                    yield joint

# Generate classifier probabilities systematically (example with S and O)
def generate_systematic_classifier_probs():
    for prob_00 in np.arange(0.05, 1.05, 0.05):
        for prob_01 in np.arange(0.05, 1.05, 0.05):
            classifier_probs = {
                (0, 0): prob_00,
                (0, 1): prob_01,
                (1, 0): 1 - prob_00,  # Assuming symmetry
                (1, 1): 1 - prob_01,  # Assuming symmetry
            }
            yield classifier_probs
            
            
from scipy.special import rel_entr

def compute_kl_y_given_e(joint_true, joint_est, classifier_probs):
    """
    Computes KL divergence between true and estimated p(y=1 | e) distributions.

    Args:
        joint_true (dict): True joint distribution, keys like 'p001'.
        joint_est (dict): Estimated joint distribution, same format.
        classifier_probs (dict): {(s, o): P(y=1 | s, o)}

    Returns:
        float: KL divergence D_KL(p_true(y|e) || p_est(y|e))
    """
    def get_p_y_given_e(joint):
        # Convert joint string keys to tuples
        parsed = {tuple(map(int, k[1:])): v for k, v in joint.items()}
        # p(e)
        p_e = {e: sum(v for (s, o, e2), v in parsed.items() if e2 == e) for e in [0, 1]}
        # p(y=1 | e)
        p_y_given_e = {}
        for e in [0, 1]:
            p_so_e = [(s, o, v / p_e[e]) for (s, o, e2), v in parsed.items() if e2 == e and p_e[e] > 0]
            p_y = sum(classifier_probs[(s, o)] * p for (s, o, p) in p_so_e)
            p_y_given_e[e] = [1 - p_y, p_y]  # [P(y=0|e), P(y=1|e)]
        return p_y_given_e

    p_true = get_p_y_given_e(joint_true)
    p_est = get_p_y_given_e(joint_est)

    kl_total = 0.0
    for e in [0, 1]:
        kl = sum(rel_entr(p_true[e], p_est[e]))
        kl_total += kl * (sum(v for k, v in joint_true.items() if int(k[-1]) == e))
    return kl_total