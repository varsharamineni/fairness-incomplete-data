
import numpy as np 
from itertools import product


# Generate all possible valid values for a 2x2 joint distribution
def generate_all_p_a(range = np.arange(0.1, 0.9, 0.1)):
    p_a_list = []
    # Loop over all possible values for p_a(x1=0, x2=0), p_a(x1=0, x2=1), p_a(x1=1, x2=0)
    for p00, p01, p10 in product(range, repeat=3):
        p11 = 1 - (p00 + p01 + p10)  # Ensure the sum equals 1
        if p11 >= 0 and (p10 + p11) > 0.5:  # Valid distribution if the last value is non-negative
            p_a = {
                (0, 0): p00,
                (1, 0): p10,
                (0, 1): p01,
                (1, 1): p11
            }
            
            
            p_a_list.append(p_a)
    return p_a_list

def generate_all_p_b(range = np.arange(0.1, 0.9, 0.1)):
    p_b_list = []
    # Loop over all possible values for p_b(x2=0, x3=0), p_b(x2=0, x3=1), p_b(x2=1, x3=0)
    for p00, p01, p10 in product(range, repeat=3):
        p11 = 1 - (p00 + p01 + p10)  # Ensure the sum equals 1
        if p11 >= 0:  # Valid distribution if the last value is non-negative
            p_b = {
                (0, 0): p00,
                (1, 0): p10,
                (0, 1): p01,
                (1, 1): p11
            }
            p_b_list.append(p_b)
    return p_b_list

# Check for marginal consistency over x2
def check_marginal_consistency(p_a, p_b, num_states = 2):
 # Marginalize p_a over x1 to get p_a(x2)
    p_a_marginal = {x2: sum(p_a[(x1, x2)] for x1 in range(num_states)) for x2 in range(num_states)}
    
    # Marginalize p_b over x3 to get p_b(x2)
    p_b_marginal = {x2: sum(p_b[(x2, x3)] for x3 in range(num_states)) for x2 in range(num_states)}
    
    # Check if the marginals over x2 are consistent
    return np.allclose(list(p_a_marginal.values()), list(p_b_marginal.values()))
