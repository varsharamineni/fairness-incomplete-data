import pandas as pd
import itertools
from collections import defaultdict
import numpy as np

def compute_distributions(data_file, s_var, o_var, e_var):
    """
    Compute marginal distributions for (o, e) and (s, o) from the given data.

    Args:
        data_file (str): Path to the CSV file containing the data.
        s_var (str): Column name for the variable representing 's' (e.g., employment-since).
        o_var (str): Column name for the variable representing 'o' (e.g., housing).
        e_var (str): Column name for the variable representing 'e' (e.g., sex).

    Returns:
        tuple: Two dictionaries containing the marginal distributions:
               - pa: Marginal distribution for (o, e)
               - pb: Marginal distribution for (s, o)
    """
    # Load the data
    df = pd.read_csv(data_file)

    # Ensure categorical data
    for col in [s_var, o_var, e_var]:
        codes, uniques = pd.factorize(df[col], sort=True)
        print(f"{col}: {uniques}")
        df[col] = codes
        df[col] = df[col].astype('category')

    # Initialize dictionaries for marginals
    pa, pb = {}, {}

    # Compute p_A(o, e): joint distribution of o and e
    for (o, e), prob in (
        df.groupby([o_var, e_var], observed=False)
        .size()
        .div(len(df))  # Normalize to get probabilities
        .items()
    ):
        pa[(o, e)] = prob

    # Compute p_B(s, o): joint distribution of s and o
    for (s, o), prob in (
        df.groupby([s_var, o_var], observed=False)
        .size()
        .div(len(df))  # Normalize to get probabilities
        .items()
    ):
        pb[(s, o)] = prob
        
        
    p_joint = {}
    
    # Compute p_SOE(s, o, e): joint distribution of s, o, and e
    # Get all possible combinations of s, o, and e
    s_values = df[s_var].cat.categories
    o_values = df[o_var].cat.categories
    e_values = df[e_var].cat.categories
    all_combinations = list(itertools.product(s_values, o_values, e_values))

    # Compute actual probabilities
    joint_probs = (
        df.groupby([s_var, o_var, e_var], observed=False)
        .size()
        .div(len(df))  # Normalize to get probabilities
    )

    # Populate p_soe with all combinations, setting missing ones to 0
    for s, o, e in all_combinations:
        key = f"p{s}{o}{e}"  # Create key in the format 'p_soe'
        p_joint[key] = joint_probs.get((s, o, e), 0.0)

    return pa, pb, p_joint



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

    # --- Step 2: Compute p(s|o) from pb ---
    pb_o = defaultdict(float)
    for (s, o), val in pb.items():
        pb_o[o] += val
        
    print(pa_o == pa_o)

    # Assuming equal contribution from both marginals (as per your formula)
    # You could also sum over pa_o and pb_o separately if you want.
    p_o = defaultdict(float)
    for o in pa_o:
        p_o[o] =  0.5 * pb_o[o] + 0.5 * pa_o[o]

    # --- Step 3: Compute p(s|o) for all combinations ---
    s_values = [0, 1]  # Binary values for s
    o_values = [0, 1]  # Binary values for o
    e_values = [0, 1]  # Binary values for e

    p_s_given_o = defaultdict(float)
    for o in o_values:
        for s in s_values:
            p_so = pb[(s, o)]
            denom = pb_o[(o, 0)]
            if denom > 0:
                p_s_given_o[(s, o)] = p_so / denom
            else:
                p_s_given_o[(s, o)] = np.nan  # undefined if p(o) == 0

    # --- Step 4: Compute p(e|o) for all combinations ---
    p_e_given_o = defaultdict(float)
    for o in o_values:
        for e in e_values:
            p_oe = pa[(o, e)]
            denom = pa_o[(o, 0)]
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


def compute_joint_marginal_preservation(pb, pa):
    """
    Compute p(s, o, e) = p(s, o) * p(e|o)
    
    Args:
        pb: dict with keys (s, o) representing p(s, o)
        pa: dict with keys (o, e) representing p(o, e)

    Returns:
        p_soe: dict with keys (s, o, e) representing joint p(s, o, e)
    """
    # Step 1: Compute p(o) from pa (sum over e)
    p_o = defaultdict(float)
    for (o, e), val in pa.items():
        p_o[o] += val

    # Step 2: Compute p(e|o) for all combinations
    e_values = [0,1]
    o_values = [0,1]

    p_e_given_o = defaultdict(float)
    for o in o_values:
        for e in e_values:
            p_oe = pa[(o, e)]
            denom = p_o[o]
            if denom > 0:
                p_e_given_o[(e, o)] = p_oe / denom
            else:
                p_e_given_o[(e, o)] = np.nan  # undefined if p(o) == 0

    # Step 3: Compute p(s, o, e) = p(s, o) * p(e|o) for all combinations
    s_values = [0, 1]  # Binary values for s

    p_soe = defaultdict(float)
    for o in o_values:
        for s in s_values:
            p_so = pb[(s, o)]
            for e in e_values:
                p_eo = p_e_given_o[(e, o)]
                p_joint = p_so * p_eo
                p_soe[f"p{s}{o}{e}"] = p_joint
                
    #print("p_soe", p_soe)

    return p_soe






