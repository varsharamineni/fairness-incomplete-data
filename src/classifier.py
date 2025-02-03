import numpy as np 
import pandas as pd 


def sample_data_from_marginal(distribution, n_samples=1000):
    """
    Samples data points from a given joint probability distribution.

    Parameters:
        joint_distribution: dict
            A joint distribution dictionary.
        n_samples: int
            Number of data points to sample.

    Returns:
        pd.DataFrame: A DataFrame with columns X1, X2, X3.
    """
    # Extract probabilities and outcomes
    outcomes = [(0, 0), (0, 1), (1, 0), (1, 1)]
    probabilities = [distribution[outcome] for outcome in outcomes]

    #probabilities = [distribution[f"p{''.join(map(str, outcome))}"] for outcome in outcomes]
    
    # Sample indices based on probabilities
    sampled_indices = np.random.choice(len(outcomes), size=n_samples, p=probabilities)
    
    # Create the dataset
    data = np.array([outcomes[i] for i in sampled_indices])
    return pd.DataFrame(data, columns=["X2", "X3"])




def classifier_to_test(data, model_type):
    
    """
    Full pipeline to sample data, fit regression models, and calculate fairness metrics.

    Parameters:
        valid_distributions: list
            List of valid joint distributions.
        n_samples: int
            Number of data points per dataset.

    Returns:
        list: Fairness metrics for each joint distribution.
    """    
    # Step 2: Train regression model
    X = data[["X2"]]
    y = data["X3"]
    model = model_type
    model.fit(X, y)
    
    return model