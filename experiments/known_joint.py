from collections import defaultdict
import numpy as np
from src.obtain_joint_list import solve_joint_distribution_iterate
from sklearn.tree import DecisionTreeClassifier
from src.get_fairness_results import generate_all_distributions_new, generate_all_distributions_new_cond
from src.visualisation import analyze_equal_opportunity
from src.obtain_joint_list import solve_joint_distribution_iterate
from src.classifier import classifier_to_test, sample_data_from_marginal


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

np.random.seed(42)  # Replace 42 with any seed number you'd like

def generate_strictly_positive_joint():
    # There are 8 combinations of (x1, x2, x3) when all are binary
    keys = [(x1, x2, x3) for x1 in [0, 1] for x2 in [0, 1] for x3 in [0, 1]]
    
    # Generate positive random probabilities
    values = np.random.uniform(0.05, 1.0, size=8)
    values /= values.sum()  # Normalize to sum to 1
    
    # Create joint distribution
    joint = {k: float(v) for k, v in zip(keys, values)}
    return joint

def get_marginals(joint):
    # Compute p_a(x1, x2)
    p_a = defaultdict(float)
    for (x1, x2, x3), prob in joint.items():
        p_a[(x1, x2)] += prob

    # Compute p_b(x2, x3)
    p_b = defaultdict(float)
    for (x1, x2, x3), prob in joint.items():
        p_b[(x2, x3)] += prob

    return dict(p_a), dict(p_b)

def break_x2_consistency(p_b, alpha=1.5, beta=1.0):
    """
    Given p_b: dict mapping (x2,x3)->prob, return a new p_b'
    where p_b'(0,*) is up‑weighted by alpha, p_b'(1,*) by beta,
    then renormalized.  This guarantees p_b'(x2) != p_a(x2).
    """
    # re‑weight
    p_new = {}
    for (x2, x3), v in p_b.items():
        p_new[(x2, x3)] = v * (alpha if x2==0 else beta)
    # renormalize total mass to 1
    total = sum(p_new.values())
    for k in p_new:
        p_new[k] /= total
    return p_new

def check_marginal_consistency(p_a, p_b, num_states=2):
    p_a_marginal = {x2: sum(p_a[(x1, x2)] for x1 in range(num_states)) for x2 in range(num_states)}
    p_b_marginal = {x2: sum(p_b[(x2, x3)] for x3 in range(num_states)) for x2 in range(num_states)}
    
    print("p_a(x2) marginal:", p_a_marginal)
    print("p_b(x2) marginal:", p_b_marginal)
    
    return np.allclose(list(p_a_marginal.values()), list(p_b_marginal.values()))

num_trials = 100 # Or however many you want
results = []

for trial in range(num_trials):
    print(f"\n--- Trial {trial + 1} ---")
    
    # 1. Generate joint
    joint = generate_strictly_positive_joint()
    
    try:
        # 3. Generate valid joint distributions
        df, true_eod = generate_all_distributions_new_cond(joint, n_samples=100,
                                            classifier=DecisionTreeClassifier(),
                                            min_value=0, max_value=1, num_joint=200)
    except Exception as e:
        print(f"❌ Skipping Trial {trial + 1} due to error during distribution generation:\n{e}")
        continue

    if df.empty or "equal_opportunity" not in df.columns:
        print(f"❌ Skipping Trial {trial + 1} due to empty DataFrame or missing column.")
        continue


    print(true_eod)
    # 4. Analyze mean EOD from df
    mean_eod = df["equal_opportunity"].mean()
    min_eod = df["equal_opportunity"].min()
    max_eod = df["equal_opportunity"].max()
    eod_range = max_eod - min_eod
    diff = true_eod - mean_eod


   
    # Store result
    results.append({
        "trial": trial + 1,
        "true_eod": true_eod,
        "mean_eod": mean_eod,
        "diff": diff,
        'eod_range': eod_range,
        'max_eod': max_eod,
        'min_eod': min_eod,
    })

    print(f"✅ Trial {trial + 1} - True EOD: {true_eod:.4f}, Mean EOD: {mean_eod:.4f}, Diff: {diff:.4f}")

# Convert results to DataFrame if needed
results_df = pd.DataFrame(results)
print("\n✅ All Successful Results:")
print(results_df)

plt.figure(figsize=(8, 5))
plt.hist(results_df["diff"], bins=15, color="mediumseagreen", edgecolor="black", alpha=0.8)
plt.axvline(x=0, color='red', linestyle='--', linewidth=1, label='Zero Line')
plt.title("Histogram of EOD Differences (True EOD - Mean EOD)")
plt.xlabel("EOD Difference")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(results_df["eod_range"], bins=15, color="mediumseagreen", edgecolor="black", alpha=0.8)
plt.xlabel("EOD Range")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

results_df['is_within_range'] = (results_df['min_eod'] <= results_df['true_eod']) & (results_df['true_eod'] <= results_df['max_eod'])
percentage_within_range = (results_df['is_within_range'].sum() / len(results_df)) * 100


plt.figure(figsize=(10, 6))

plt.scatter(results_df.index, results_df['true_eod'], color='blue', label='True EOD', zorder=5)

lower_error = results_df['true_eod'] - results_df['min_eod']
upper_error = results_df['max_eod'] - results_df['true_eod']


plt.plot(results_df.index, results_df['min_eod'], 'g--', label='min_eod', zorder=3)
plt.plot(results_df.index, results_df['max_eod'], 'r--', label='max_eod', zorder=3)

plt.scatter(results_df.index[~results_df['is_within_range']], results_df['true_eod'][~results_df['is_within_range']], color='red', label='Out of Range', zorder=6)
plt.text(0.5, 0.8, f'Percentage within bounds: {percentage_within_range:.2f}%', fontsize=12, ha='center', va='center', transform=plt.gca().transAxes, color='green')

plt.xlabel('Index')
plt.ylabel('EOD Value')
plt.title('True EOD and Range Check')
plt.legend()

plt.tight_layout()
plt.show()