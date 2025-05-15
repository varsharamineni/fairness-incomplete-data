import numpy as np
import math

def perturb_pb(p_b, noise_std=0.05):
    perturbed_pb = {key: value + np.random.normal(0, noise_std) for key, value in p_b.items()}
    total = sum(perturbed_pb.values())
    return {key: value / total for key, value in perturbed_pb.items()}

def break_o_consistency(p_b, alpha=1.5, beta=1.0):
    p_new = {}
    for (s, o), v in p_b.items():
        # Now scaling based on x3 (the o variable)
        p_new[(s, o)] = v * (alpha if o == 0 else beta)
    
    # Normalize the distorted distribution
    total = sum(p_new.values())
    for k in p_new:
        p_new[k] /= total
        
    return p_new

def check_marginal_consistency_kl(pa, pb):
    """
    Check marginal consistency between p_a(o) and p_b(o) using KL divergence.
    
    Args:
        pa (dict): The marginal p_a(o, e).
        pb (dict): The marginal p_b(s, o).
        
    Returns:
        float: KL divergence between p_a(o) and p_b(o). 0 means perfect consistency.
    """
    # Compute marginal p_a(o) and p_b(o)
    pa_o = {}
    pb_o = {}

    for (o, e), value in pa.items():
        pa_o[o] = pa_o.get(o, 0) + value

    for (s, o), value in pb.items():
        pb_o[o] = pb_o.get(o, 0) + value

    # Normalize to ensure valid probability distributions
    total_pa = sum(pa_o.values())
    total_pb = sum(pb_o.values())
    for o in pa_o:
        pa_o[o] /= total_pa
    for o in pb_o:
        pb_o[o] /= total_pb

    # Compute KL divergence D_KL(pa_o || pb_o)
    kl_div = 0.0
    epsilon = 1e-12  # to avoid log(0)
    for o in pa_o:
        p = pa_o[o]
        q = pb_o.get(o, epsilon)
        if p > 0:
            kl_div += p * math.log(p / q)

    return kl_div