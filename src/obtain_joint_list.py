import numpy as np


def solve_joint_distribution_iterate(pa, pb, num=50):
    
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
                
                # Ensure all probabilities are positive and not zero
                #if any(value <= 0.01 for value in joint_distribution.values()):
                #    continue  # Skip this combination if any probability is <= 0
                
                # Ensure all constraints are met
                total_sum = sum(joint_distribution.values())
                all_positive = all(value >= 0 for value in joint_distribution.values())
                
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
                
                # If all conditions are satisfied, add to the list
                if abs(total_sum - 1) < 1e-9 and all_positive and pa_valid and pb_valid:
                    valid_joint_distributions.append(joint_distribution)
            
            except AssertionError:
                # Ignore invalid combinations
                continue
    
    return valid_joint_distributions



