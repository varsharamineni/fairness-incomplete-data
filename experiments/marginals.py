from collections import defaultdict
import numpy as np
from src.obtain_joint_list import solve_joint_distribution_iterate
from sklearn.tree import DecisionTreeClassifier
from src.get_fairness_results import generate_all_distributions_new, equal_opportunity_difference
from src.visualisation import analyze_equal_opportunity
from src.obtain_joint_list import solve_joint_distribution_iterate
from src.classifier import classifier_to_test, sample_data_from_marginal


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



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

def check_marginal_consistency(p_a, p_b, num_states=2):
    p_a_marginal = {x2: sum(p_a[(x1, x2)] for x1 in range(num_states)) for x2 in range(num_states)}
    p_b_marginal = {x2: sum(p_b[(x2, x3)] for x3 in range(num_states)) for x2 in range(num_states)}
    
    print("p_a(x2) marginal:", p_a_marginal)
    print("p_b(x2) marginal:", p_b_marginal)
    
    return np.allclose(list(p_a_marginal.values()), list(p_b_marginal.values()))

# Example usage
joint = generate_strictly_positive_joint()
p_a, p_b = get_marginals(joint)

print("Are marginals consistent over x2?", check_marginal_consistency(p_a, p_b))

print("\nJoint Distribution:")
for k, v in joint.items():
    print(f"{k}: {v:.4f}")
    
# Solve for valid joint distributions
df = generate_all_distributions_new(p_a, p_b, n_samples=100, classifier=DecisionTreeClassifier(), min_value = 0.02, max_value = 1, num_joint=200)


analyze_equal_opportunity(df, show_plots=False, verbose=True)


print(joint)
PAXY = {f'p{key[0]}{key[1]}{key[2]}': value for key, value in joint.items()}

# Sample training data and train model
train_data_sample = sample_data_from_marginal(p_b, n_samples=100)
print(train_data_sample)
model = classifier_to_test(train_data_sample, DecisionTreeClassifier())
classifier_probs = {
x: model.predict_proba(pd.DataFrame([[x]], columns=['X2']))[0][1]
for x in [0, 1]
}
print(classifier_probs)

eod = equal_opportunity_difference(PAXY, classifier_probs)
print(eod)



mean_eod = df["equal_opportunity"].mean()
diff = eod - mean_eod


plt.figure(figsize=(8, 5))
plt.hist(df['equal_opportunity'], bins=20, color='skyblue', edgecolor='black')
plt.axvline(x=eod, color='red', linestyle='--', label='Threshold = 0.1')
plt.title('Distribution of Equal Opportunity Difference (EOD)')
plt.xlabel('Equal Opportunity Difference')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()





#print(joint)