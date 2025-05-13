import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from src.get_fairness_results import generate_all_distributions_new, equal_opportunity_difference
from src.visualisation import analyze_equal_opportunity
from src.obtain_joint_list import solve_joint_distribution_iterate


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



print("Script started")  # Debugging line

import numpy as np
import pandas as pd

def generate_fake_data(n=1000, seed=500):
    """
    Generates a fake dataset with three binary variables, each with different probabilities for 0 or 1.
    """
    np.random.seed(seed)
    
    # Define different probabilities for each column
    p_X1 = [0.7, 0.3]  # Probability of 0 and 1 for X1
    p_X2 = [0.8, 0.2]  # Probability of 0 and 1 for X2
    p_X3 = [0.6, 0.4]  # Probability of 0 and 1 for X3
    
    # Generate data with different probabilities for each column
    X1 = np.random.choice([0, 1], size=n, p=p_X1)
    X2 = np.random.choice([0, 1], size=n, p=p_X2)
    X3 = np.random.choice([0, 1], size=n, p=p_X3)
    
    # Combine the columns into a DataFrame
    df = pd.DataFrame({"X1": X1, "X2": X2, "X3": X3})
    
    return df

def compute_marginals(df):
    """
    Computes marginal distributions P(X1, X2) and P(X2, X3) from the dataset.
    """
    pa = df.groupby(["X1", "X2"]).size().div(len(df)).to_dict()
    pb = df.groupby(["X2", "X3"]).size().div(len(df)).to_dict()
    return pa, pb

def compute_joint(df):
    """
    Computes marginal distributions P(X1, X2) and P(X2, X3) from the dataset.
    """
    p_joint = df.groupby(["X1", "X2", "X3"]).size().div(len(df)).to_dict()
    return p_joint

# Check for marginal consistency over x2
def check_marginal_consistency(p_a, p_b, num_states = 2):
 # Marginalize p_a over x1 to get p_a(x2)
    p_a_marginal = {x2: sum(p_a[(x1, x2)] for x1 in range(num_states)) for x2 in range(num_states)}
    
    # Marginalize p_b over x3 to get p_b(x2)
    p_b_marginal = {x2: sum(p_b[(x2, x3)] for x3 in range(num_states)) for x2 in range(num_states)}
    
    # Check if the marginals over x2 are consistent
    return np.allclose(list(p_a_marginal.values()), list(p_b_marginal.values()))

# Generate fake dataset
df = generate_fake_data(n=10000)

print(df)

# Compute marginal distributions
p_a, p_b = compute_marginals(df)

print(p_a)
print(p_b)

print(check_marginal_consistency(p_a, p_b))

print(compute_joint(df))

# Ensure all keys are present in the marginals
keys = [(0,0), (1,0), (0,1), (1,1)]
for key in keys:
    p_a.setdefault(key, 0)
    p_b.setdefault(key, 0)
    
    
    
    
p_a = {(0, 0): 0.5, (0, 1): 0.1, (1, 0): 0.2, (1, 1): 0.1}
p_b = {(0, 0): 0.5, (0, 1): 0.2, (1, 0): 0.1, (1, 1): 0.1}
print(check_marginal_consistency(p_a, p_b))


joint = solve_joint_distribution_iterate(p_a, p_b, min_value=0, max_value=1, num=100)  

print(joint)

# Solve for valid joint distributions
#df = generate_all_distributions_new(p_a, p_b, n_samples=100, classifier=DecisionTreeClassifier(), min_value = 0, max_value = 1, num_joint=200)


#analyze_equal_opportunity(df, show_plots=False, verbose=True)
