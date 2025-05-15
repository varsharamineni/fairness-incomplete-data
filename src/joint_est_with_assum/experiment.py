import pandas as pd
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 


def compute_fairness(y_pred, protected_attr):
    df = pd.DataFrame({'y_pred': y_pred, 'protected': protected_attr})
    groups = df.groupby('protected')['y_pred'].mean()
    
    if len(groups) != 2:
        raise ValueError("Protected attribute must have exactly 2 groups to compute DI and DD")

    # Access means by group label explicitly (0 = unpriv, 1 = priv)
    p_unpriv = groups.loc[0]
    p_priv = groups.loc[1]
    
    # DI = P(y_hat=1 | unpriv) / P(y_hat=1 | priv)
    di = p_unpriv / p_priv if p_priv != 0 else float('inf')
    
    # DD = difference of probabilities
    dd = p_unpriv - p_priv
    
    return di, dd


def evaluate_fairness(df_test, feature_cols, prot_attr_col, classifier=None):
    """
    Evaluate fairness (DI, DD) using predictions based on feature_cols and protected attribute.
    
    Args:
        df: pd.DataFrame containing test data.
        feature_cols: list of str, predictive feature columns used by the model.
        prot_attr_col: str, column name of the protected attribute.
        classifier: fitted model (e.g. sklearn), optional.
        y_proba: numpy array of predicted probabilities (optional if classifier is passed).
        threshold: float, threshold for converting probabilities to binary predictions.

    Returns:
        Dictionary with fairness metrics.
    """
    X_test = df_test[feature_cols]
    protected_attr = df_test[prot_attr_col]

    if classifier is not None:
        y_pred = classifier.predict(X_test)

    di, dd = compute_fairness(y_pred, protected_attr)

    print(f"Disparate Impact (DI): {di:.3f}")
    print(f"Demographic Disparity (DD): {dd:.3f}")

    return {'DI': di, 'DD': dd}




def run_fairness_experiment(
    clf, 
    real_test_df, 
    synth_test_df, 
    pred_features, 
    protected_attr,
    pred_col_name='prediction'
):
    """
    Given a trained classifier and real/synthetic test data,
    calculates and returns fairness metrics on both datasets.
    
    Parameters:
    - clf: trained classifier with .predict or .predict_proba method
    - real_test_df: pd.DataFrame with test data (real)
    - synth_test_df: pd.DataFrame with test data (synthetic)
    - pred_features: list of columns to use for prediction
    - protected_attr: string, name of protected attribute column in data
    - pred_col_name: string, name for prediction column to create
    
    Returns:
    - dict with fairness metrics on real and synthetic data
    """
    
    # Predict on real test data
    X_real = pd.get_dummies(real_test_df[pred_features])
    real_test_df[pred_col_name] = clf.predict(X_real)
    
    # Predict on synthetic test data
    X_synth = pd.get_dummies(synth_test_df[pred_features])
    synth_test_df[pred_col_name] = clf.predict(X_synth)
    
    # Calculate fairness metrics
    real_metrics = compute_fairness(real_test_df[pred_col_name], real_test_df[protected_attr])
    synth_metrics = compute_fairness(synth_test_df[pred_col_name], synth_test_df[protected_attr])
    
    results = {
        'real': real_metrics,
        'synthetic': synth_metrics
    }
    
    return results



np.random.seed(0)
df_test = pd.DataFrame({
    's': np.random.choice([0, 1], size=1000, p=[0.7, 0.3]),
    'o': np.random.choice([0, 1], size=1000, p=[0.4, 0.6]),
    'e': np.random.choice([0, 1], size=1000, p=[0.5, 0.5]),
    'target': np.random.choice([0,1], size=1000, p=[0.6, 0.4])  # separate target for classifier
})


X = df_test.drop(columns=['e', 'target'])
y = df_test['target']

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
clf = DecisionTreeClassifier().fit(X_train, y_train)

# Evaluate fairness on real test data
print("Fairness on real test data:")
evaluate_fairness(df_test.iloc[X_val.index], X.columns.tolist(), prot_attr_col='e', classifier=clf)

# Create synthetic test data similar to df_test (for demonstration)
synth_test_df = df_test.sample(n=200, replace=True).reset_index(drop=True)

# Run fairness experiment comparing real vs synthetic
results = run_fairness_experiment(
    clf=clf, 
    real_test_df=df_test.iloc[X_val.index], 
    synth_test_df=synth_test_df,
    pred_features=['s', 'o'],  # raw categorical; will be one-hot encoded inside function
    protected_attr='e'
)

print("\nFairness metrics comparison (real vs synthetic):")
print(results)