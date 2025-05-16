from joint_feasible_set.experiment_real_data import run_full_experiment_real_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


pd.set_option('display.max_rows', None)   # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns


def train_tree_and_get_probs(X, y, random_state=None):
    """
    Trains a Decision Tree classifier to estimate p(y = 1 | s, o).
    
    Args:
        X (pd.DataFrame): Feature matrix with columns [s, o].
        y (array-like): Binary target variable (0 or 1).
        random_state (int or None): Seed for reproducibility.
        
    Returns:
        dict: {(s, o): p(y=1 | s, o)} for all observed (s, o) pairs.
    """
    clf = DecisionTreeClassifier(random_state=random_state)
    clf.fit(X, y)

    classifier_probs = {}

    s_values = X.iloc[:, 0].unique()
    o_values = X.iloc[:, 1].unique()

    y_proba = clf.predict_proba(X)
    y1_col = list(clf.classes_).index(1)  # column corresponding to y=1

    for s in s_values:
        for o in o_values:
            mask = (X.iloc[:, 0] == s) & (X.iloc[:, 1] == o)
            if mask.sum() == 0:
                classifier_probs[(s, o)] = 0.5  # default fallback
            else:
                prob = np.mean(y_proba[mask, y1_col])
                classifier_probs[(s, o)] = prob

    return classifier_probs

def load_data_from_folder(file_path):
    data = pd.read_csv(file_path)
    return data


alphas = [1.0] # Alpha values for consistency-breaking
betas =  [1.0]  # Beta values for consistency-breaking
num_candidates = 100     # Number of candidates to reconstruct
e = 'sex'
o = 'housing'
s = 'employment-since'
label = 'class-label'  # Label for the target variable
data_folder = 'real_data/processed/german_clean.csv'

df = load_data_from_folder(data_folder)
classifier_probs = train_tree_and_get_probs(df[[s, o]], df[label], random_state=20)


if __name__ == "__main__":
    # Set experiment parameters
    # Run the full experiment
    results_df = run_full_experiment_real_data(
        data_folder = data_folder,
        s_var=s,
        o_var=o,
        e_var=e,
        alphas=alphas,
        betas=betas,
        num_candidates=num_candidates,
        seed=502,
        classifier_probs=classifier_probs
    )

    # Save the results to a CSV file
    results_df.to_csv("results/real_data/german_data_di_dd_kl.csv", index=False)
    print("Experiment completed'.")
    print(results_df)  # Print the first few rows of results for verification
    
    
    
    
   