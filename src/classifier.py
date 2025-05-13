import numpy as np 
import pandas as pd 


def sample_data_from_marginal(distribution, n_samples=1000):
    """
    Samples data points from a given joint probability distribution.

    Parameters:
        distribution: dict
            A joint distribution dictionary.
        n_samples: int
            Number of data points to sample.

    Returns:
        pd.DataFrame: A DataFrame with columns X2, X3.
    """
    # Extract probabilities and outcomes
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    probabilities = [distribution[outcome] for outcome in outcomes]
    
    # Sample indices based on probabilities
    sampled_indices = np.random.choice(len(outcomes), size=n_samples, p=probabilities)
    
    # Create the dataset
    data = np.array([outcomes[i] for i in sampled_indices])
    return pd.DataFrame(data, columns=["X2", "X3"])


def classifier_to_test(data, model):
    """
    Fit a classification model using the provided data.

    Parameters:
        data (pd.DataFrame): 
            DataFrame containing the dataset with features and labels.
            Must include 'X2' as a feature and 'X3' as the target.
        model (sklearn.base.ClassifierMixin): 
            A scikit-learn compatible classification model instance.

    Returns:
        sklearn.base.ClassifierMixin: 
            The trained classification model.
    """
    X = data[['X2']]
    y = data['X3']
    model.fit(X, y)
    
    return model