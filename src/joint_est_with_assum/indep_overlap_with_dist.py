import pandas as pd
import itertools
from collections import defaultdict
import numpy as np



def compute_joint_indep_given_overlap(pa, pb):
    """
    Compute joint p(s, o, e) = p(o) * p(s|o) * p(e|o)
    - pa: dict of (o, e) -> p(o, e)
    - pb: dict of (s, o) -> p(s, o)
    
    Returns:
        p_soe: dict with keys like 'p001' representing p(s, o, e)
    """
    # --- Step 1: Compute p(o) from pa (sum over e) ---
    pa_o = defaultdict(float)
    for (o, e), val in pa.items():
        pa_o[o] += val
        
    print("pa_o", pa_o)

    # --- Step 2: Compute p(s|o) from pb ---
    pb_o = defaultdict(float)
    for (s, o), val in pb.items():
        pb_o[o] += val
        
    print("pb_o", pb_o)
    
    # --- Step 3: Compute p(s|o) for all combinations ---
    s_values = [0, 1]  # Binary values for s
    o_values = [0, 1]  # Binary values for o
    e_values = [0, 1]  # Binary values for e
        
    # Assuming equal contribution from both marginals (as per your formula)
    # You could also sum over pa_o and pb_o separately if you want.
    p_o = defaultdict(float)
    for o in o_values:
        p_o[o] =  (0.5 * pb_o[o]) + (0.5 * pa_o[o])



    p_s_given_o = defaultdict(float)
    for o in o_values:
        for s in s_values:
            p_so = pb[(s, o)]
            denom = pb_o[o]
            if denom > 0:
                p_s_given_o[(s, o)] = p_so / denom
            else:
                p_s_given_o[(s, o)] = np.nan  # undefined if p(o) == 0

    # --- Step 4: Compute p(e|o) for all combinations ---
    p_e_given_o = defaultdict(float)
    for o in o_values:
        for e in e_values:
            p_oe = pa[(e, o)]
            denom = pa_o[o]
            if denom > 0:
                p_e_given_o[(e, o)] = p_oe / denom
            else:
                p_e_given_o[(e, o)] = np.nan  # undefined if p(o) == 0

    # --- Step 5: Combine to get p(s, o, e) for all combinations ---
    p_soe = {}

    # Iterate over all combinations of s, o, e (since they are binary, the range is [0, 1])
    for o in o_values:
        for s in s_values:
            for e in e_values:
                # Access p(s|o) and p(e|o), and calculate the joint probability
                ps_given_o = p_s_given_o[(s, o)]
                pe_given_o = p_e_given_o[(e, o)]
                p_joint = p_o[o] * ps_given_o * pe_given_o
                p_soe[f"p{s}{o}{e}"] = p_joint
                
    return p_soe









