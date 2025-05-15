import pandas as pd
import itertools
from collections import defaultdict
import numpy as np

def compute_joint_marginal_preservation(pa, pb):
    """
    Compute p(s, o, e) = p(e, o) * p(s|o)
    
    Args:
        pb: dict with keys (s, o) representing p(s, o)
        pa: dict with keys (o, e) representing p(o, e)

    Returns:
        p_soe: dict with keys (s, o, e) representing joint p(s, o, e)
    """
    # --- Step 1: Compute p(o) from pa (sum over e) ---
    pa_o = defaultdict(float)
    for (o, e), val in pa.items():
        pa_o[o] += val
        
    # --- Step 2: Compute p(s|o) from pb ---
    pb_o = defaultdict(float)
    for (s, o), val in pb.items():
        pb_o[o] += val
        
    # --- Step 3: Compute p(s|o) for all combinations ---
    s_values = [0, 1]  # Binary values for s
    o_values = [0, 1]  # Binary values for o
    e_values = [0, 1]  # Binary values for e

    p_s_given_o = defaultdict(float)
    for o in o_values:
        for s in s_values:
            p_so = pb[(s, o)]
            denom = pb_o[o]
            if denom > 0:
                p_s_given_o[(s, o)] = p_so / denom
            else:
                p_s_given_o[(s, o)] = np.nan  # undefined if p(o) == 0

    # --- Step 5: Combine to get p(s, o, e) for all combinations ---
    p_soe = {}
    
    # Iterate over all combinations of s, o, e (since they are binary, the range is [0, 1])
    for o in o_values:
        for s in s_values:
            for e in e_values:
                ps_given_o = p_s_given_o[(s, o)]
                poe = pa[(e, o)]
                p_joint = ps_given_o * poe
                p_soe[f"p{s}{o}{e}"] = p_joint
                
    return p_soe