import numpy as np
import pandas as pd
import itertools

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