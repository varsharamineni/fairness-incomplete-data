import pandas as pd
from joint_est_with_assum.em_algorithm import em_algo, compute_joint_em

# Your datasets (make sure values are 0-indexed!)
df1 = pd.DataFrame({
    's': [0, 1, 0, 1],
    'o': [1, 0, 1, 0]
})

df2 = pd.DataFrame({
    'o': [1, 0, 1, 1],
    'e': [0, 1, 1, 0]
})

# Category sizes: A, B, and C each have 2 values (0 and 1)
M = [2, 2, 2]
labels = ['s', 'o', 'e']

# Run EM
log_likelihood, final_params = em_algo(df1, df2, M, labels, K=3)

# Output
print("Final log-likelihood:", log_likelihood[-1])
print("Final parameters:")
for i, param in enumerate(final_params[:-1]):
    print(f"Variable {labels[i]}:\n{param}")
print("Mixing proportions (pi):", final_params[-1])

joint_dist = compute_joint_em(final_params, M, K=3)
print("Joint distribution:")
for key, value in joint_dist.items():
    print(f"{key}: {value}")