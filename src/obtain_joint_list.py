import numpy as np


def solve_joint_distribution_iterate(pa, pb, min_value = 0, max_value = 1, num=50):
    
    """
    Iterates over all possible values of c and k within their ranges
    to solve the joint probability distribution.

    Parameters:
        pa: dict
            Marginal probabilities
        pb: dict
            Marginal probabilities 
        num: float
            number of points for iterating over c and k.

    Returns:
        list: A list of valid joint distributions that satisfy all constraints.
    """
    # Extract marginal probabilities
    pa00, pa10, pa01, pa11 = pa[(0, 0)], pa[(1, 0)], pa[(0, 1)], pa[(1, 1)]
    pb00, pb10, pb01, pb11 = pb[(0, 0)], pb[(1, 0)], pb[(0, 1)], pb[(1, 1)]

    
    # Bounds for c
    lower_bound_c = max(0, pb00 - pa10)
    upper_bound_c = min(1, pa00, pb00)
    
    # Bounds for k
    lower_bound_k = max(0, pb10 - pa11)
    upper_bound_k = min(1, pa01, pb10)
    
    # List to store valid joint distributions
    valid_joint_distributions = []
    
    # Iterate over all possible values of c and k
    c_values = np.linspace(lower_bound_c, upper_bound_c, num)
    k_values =  np.linspace(lower_bound_k, upper_bound_k, num)

    
    for c in c_values:
        for k in k_values:
            try:
                # Compute joint probabilities
                p000 = c
                p001 = pa00 - c
                p100 = pb00 - c
                p101 = pa10 - pb00 + c
                p010 = k
                p011 = pa01 - k
                p110 = pb10 - k
                p111 = pa11 - pb10 + k

                joint_distribution = {
                    "p000": p000, "p001": p001, "p100": p100, "p101": p101,
                    "p010": p010, "p011": p011, "p110": p110, "p111": p111
                }
                #print(joint_distribution)
                
                # Ensure all probabilities are positive and not zero
                #if any(value <= 0.01 for value in joint_distribution.values()):
                #    continue  # Skip this combination if any probability is <= 0
                
                # Ensure all constraints are met
                total_sum = sum(joint_distribution.values())
                all_positive = all(value >= 0 for value in joint_distribution.values())
                #print(abs(total_sum - 1) < 1e-9 )
                #print(all_positive)
                
                # Marginal constraints
                pa_computed = {
                    (0, 0): p000 + p001,
                    (1, 0): p100 + p101,
                    (0, 1): p010 + p011,
                    (1, 1): p110 + p111,
                }
                pb_computed = {
                    (0, 0): p000 + p100,
                    (1, 0): p001 + p101,
                    (0, 1): p010 + p110,
                    (1, 1): p011 + p111,
                }
                pa_valid = all(abs(pa[key] - pa_computed[key]) < 1e-9 for key in pa)
                pb_valid = all(abs(pb[key] - pb_computed[key]) < 1e-9 for key in pb)
                
                # extra constraints
                min_threshold = all(value >= min_value for value in joint_distribution.values())
                max_threshold = all(value <= max_value for value in joint_distribution.values())
                
                                
                # If all conditions are satisfied, add to the list
                if abs(total_sum - 1) < 1e-9 and all_positive and pa_valid and pb_valid and min_threshold and max_threshold:
                    valid_joint_distributions.append(joint_distribution)
            
            except AssertionError:
                # Ignore invalid combinations
                continue
            
            #print(valid_joint_distributions)
                
    return valid_joint_distributions





def solve_joint_distribution_iterate_cond(pa, pb, min_value=0.0, max_value=1.0, num=50):
    """
    Solves for consistent joint distributions p(s, o, e) using symbolic keys like 'p011' instead of tuples.

    Parameters:
        pa: dict
            Joint distribution \hat{p}_a(o, e), with keys (o, e)
        pb: dict
            Joint distribution \hat{p}_b(s, o), with keys (s, o)
        min_value: float
            Minimum value allowed in the joint distribution
        max_value: float
            Maximum value allowed in the joint distribution
        num: int
            Number of grid values to iterate over for free parameters c and k

    Returns:
        list of dicts: Valid joint distributions with symbolic keys
    """
    # Compute p_bank(s | o)
    p_bank_cond = {}
    for o in [0, 1]:
        total = pb[(0, o)] + pb[(1, o)]
        if total == 0:
            raise ValueError(f"Total probability for o={o} in pb is zero.")
        for s in [0, 1]:
            p_bank_cond[(s, o)] = pb[(s, o)] / total

    # Extract required marginals
    pa00, pa01 = pa[(0, 0)], pa[(0, 1)]
    pa10, pa11 = pa[(1, 0)], pa[(1, 1)]
    pb00, pb01 = pb[(0, 0)], pb[(0, 1)]
    pb10, pb11 = pb[(1, 0)], pb[(1, 1)]

    p_b_0_given_0 = p_bank_cond[(0, 0)]
    p_b_0_given_1 = p_bank_cond[(0, 1)]


    # Bounds for c and k
    c_lo = max(0, pa00 - 1, (p_b_0_given_0 * (pa00 + pa01)) - 1)
    c_hi = min(1, pa00, p_b_0_given_0 * (pa00 + pa01))

    k_lo = max(0, pa10 - 1, (p_b_0_given_1 * (pa10 + pa11)) - 1)
    k_hi = min(1, pa10, p_b_0_given_1 * (pa10 + pa11))

    c_values = np.linspace(c_lo, c_hi, num)
    k_values = np.linspace(k_lo, k_hi, num)

    valid_distributions = []

    for c in c_values:
        for k in k_values:
            try:
                p000 = c
                p010 = k
                p100 = pa00 - c
                p110 = pa10 - k
                p001 = (p_b_0_given_0 * (pa00 + pa01)) - c
                p101 = pa01 - p001
                p011 = (p_b_0_given_1 * (pa10 + pa11)) - k
                p111 = pa11 - p011

                joint = {
                    'p000': p000, 'p001': p001,
                    'p010': p010, 'p011': p011,
                    'p100': p100, 'p101': p101,
                    'p110': p110, 'p111': p111
                }

                total_sum = sum(joint.values())
                all_positive = all(value >= 0 for value in joint.values())

                # Reconstruct p(s | o) from joint
                def cond_p(s, o):
                    denom = sum(joint[f"p{s_}{o}{e}"] for s_ in [0, 1] for e in [0, 1])
                    numer = sum(joint[f"p{s}{o}{e}"] for e in [0, 1])
                    return numer / denom if denom > 0 else 0.0

                p_s_o_est = {(s, o): cond_p(s, o) for s in [0, 1] for o in [0, 1]}
                p_s_o_true = {(s, o): p_bank_cond[(s, o)] for s in [0, 1] for o in [0, 1]}
                matches_cond = all(abs(p_s_o_est[k] - p_s_o_true[k]) < 1e-9 for k in p_s_o_est)
                
                # Marginal constraints
                pa_computed = {
                    (0, 0): p100 + p000,
                    (1, 0): p110 + p010,
                    (0, 1): p001 + p101,
                    (1, 1): p011 + p111,
                }
                
                pa_valid = all(abs(float(pa[key]) - pa_computed[key]) < 1e-9 for key in pa)
                
                # extra constraints
                min_threshold = all(value >= min_value for value in joint.values())
                max_threshold = all(value <= max_value for value in joint.values())

                if abs(total_sum - 1) < 1e-9 and all_positive and matches_cond and pa_valid and min_threshold and max_threshold:
                    valid_distributions.append(joint)

            except Exception:
                continue

    return valid_distributions






