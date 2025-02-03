import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from src.get_fairness_results import generate_all_distributions


print("Script started")  # Debugging line

df = generate_all_distributions(p_a_range = np.arange(0.1, 0.9, 0.1), p_b_range = np.arange(0.1, 0.9, 0.1), 
                               n_samples=100, classifier=DecisionTreeClassifier(), num_joint=50)

print(df)

print('hint')