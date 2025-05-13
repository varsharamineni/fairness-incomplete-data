
from .classifier import classifier_to_test, sample_data_from_marginal
from .generate_marginal_pairs import generate_all_p_a, generate_all_p_b, check_marginal_consistency
from .obtain_joint_list import solve_joint_distribution_iterate, solve_joint_distribution_iterate_cond

from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np 
from collections import defaultdict



## calculate metric
def compute_conditional_probability(P_joint, classifier_probs, A_value):
    
    """
    Computes the conditional probability P(Ŷ = 1 | A = a, Y = 1), where:
    - P_joint is the joint distribution over (A, X, Y),
    - classifier_probs provides P(Ŷ = 1 | X),
    - A_value is the fixed value of A to condition on.

    Parameters:
        P_joint (dict): 
            A joint distribution over tuples encoded as string keys (e.g., 'AXY'),
            where each value is P(A=a, X=x, Y=y).
        classifier_probs (dict): 
            A dictionary mapping values of X to P(Ŷ = 1 | X), as predicted by a classifier.
        A_value (int): 
            The specific value of A to condition on.

    Returns:
        float: The conditional probability P(Ŷ = 1 | A = a, Y = 1).
    """
    
    numerator = 0.0
    denominator = 0.0
    
    for key, P_AXY in P_joint.items():
        A, X, Y = int(key[1]), int(key[2]), int(key[3])  # Extract values from string
        
        if A == A_value and Y == 1:
            P_Yhat_given_X = classifier_probs[X]  # P(Ŷ = 1 | X)
            numerator += P_AXY * P_Yhat_given_X
            denominator += P_AXY
    
    print('hello   ')  
    print(numerator)
    
    return numerator / denominator if denominator > 0 else 0


def equal_opportunity_difference(P_joint, classifier_probs):
    """
    Computes Equal Opportunity Difference:
    
    EOD = P(Ŷ = 1 | Y = 1, A = 0) - P(Ŷ = 1 | Y = 1, A = 1)
    """
    P_Yhat_given_Y1_A0 = compute_conditional_probability(P_joint, classifier_probs, A_value=0)
    P_Yhat_given_Y1_A1 = compute_conditional_probability(P_joint, classifier_probs, A_value=1)
        
    return P_Yhat_given_Y1_A0 - P_Yhat_given_Y1_A1


# Function to flatten dictionary column into separate columns
def flatten_dict_column(df, column_name):
    """
    This function flattens a dictionary column in a DataFrame into individual columns.
    Each key of the dictionary becomes a column in the DataFrame.
    """
    # Normalize the dictionary column into individual columns
    dict_data = pd.json_normalize(df[column_name])
    
    # Rename columns to match the format of the dictionary keys
    dict_data.columns = [f"{column_name}_{str(key)}" for key in dict_data.columns]
    
    return dict_data



## get results

def generate_all_distributions(p_a_range = np.arange(0.1, 0.9, 0.1), p_b_range = np.arange(0.1, 0.9, 0.1), 
                               n_samples=100, classifier=DecisionTreeClassifier(), min_value = 0, max_value = 1, num_joint=50):
    """
    Generate all valid joint distributions, train models, and compute fairness metrics.
    
    Parameters:
        p_a_list (list): List of marginal probabilities for group A.
        p_b_list (list): List of marginal probabilities for group B.
        n_samples (int): Number of samples to draw for each distribution.
        classifier: ML model to use for fairness pipeline.
        num_joint (int): Number of joint distributions to generate.
    
    Returns:
        pd.DataFrame: DataFrame containing results for each valid distribution.
    """
    
    p_a_list = generate_all_p_a(p_a_range)
    p_b_list = generate_all_p_b(p_b_range)

    # Dictionary to store all valid distributions for each (p_a, p_b)
    data = []
    
    for p_a in p_a_list:
        for p_b in p_b_list:
            if check_marginal_consistency(p_a, p_b):
                
                # Sample training data and train model
                train_data_sample = sample_data_from_marginal(p_b, n_samples=n_samples)
                model = classifier_to_test(train_data_sample, classifier)
                classifier_probs = {x: model.predict_proba(np.array([[x]]))[0][1] for x in [0,1]}

                # Generate joint distributions
                joint_list = solve_joint_distribution_iterate(p_a, p_b, min_value, max_value, num=num_joint)
                
                for joint in joint_list:
                    
                    # Compute mean of fairness metrics
                    eod = equal_opportunity_difference(joint, classifier_probs)
                    
                    # Append results to data
                    data.append({
                        'p_a': p_a,
                        'p_b': p_b,
                        'joint_distribution': joint,
                        'equal_opportunity': eod
                    })
                    
    data = pd.DataFrame(data)
                    
    # Flatten the 'p_a', 'p_b', and 'joint_distribution' columns
    flattened_p_a = flatten_dict_column(data, 'p_a')
    flattened_p_b = flatten_dict_column(data, 'p_b')
    flattened_joint = flatten_dict_column(data, 'joint_distribution')

    # Step 2: Concatenate the flattened columns with the original DataFrame
    df_final = pd.concat([data.drop(['p_a', 'p_b', 'joint_distribution'], axis=1), flattened_p_a, flattened_p_b, flattened_joint], axis=1)

    #df_final.dropna(inplace=True, ignore_index=True)
    
    # Convert to DataFrame
    return df_final



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



def generate_all_distributions_new(joint, 
                               n_samples=100, classifier=DecisionTreeClassifier(), min_value = 0, max_value = 1, num_joint=50):
    """
    Generate all valid joint distributions, train models, and compute fairness metrics.
    
    Parameters:
        p_a_list (list): List of marginal probabilities for group A.
        p_b_list (list): List of marginal probabilities for group B.
        n_samples (int): Number of samples to draw for each distribution.
        classifier: ML model to use for fairness pipeline.
        num_joint (int): Number of joint distributions to generate.
    
    Returns:
        pd.DataFrame: DataFrame containing results for each valid distribution.
    """
    
    # 2. Compute marginals
    p_a, p_b = get_marginals(joint)


    # Dictionary to store all valid distributions for each (p_a, p_b)
    data = []
    
    # Sample training data and train model
    train_data_sample = sample_data_from_marginal(p_b, n_samples=n_samples)
    model = classifier_to_test(train_data_sample, classifier)
    classifier_probs = {
    x: model.predict_proba(pd.DataFrame([[x]], columns=['X2']))[0][1]
    for x in [0, 1]}
    print(classifier_probs)
    #classifier_probs = {x: model.predict_proba(np.array([[x]]))[0][1] for x in [0,1]}

    PAXY = {f'p{key[0]}{key[1]}{key[2]}': value for key, value in joint.items()}
    true_eod = equal_opportunity_difference(PAXY, classifier_probs)
    print(true_eod)



    # Generate joint distributions
    joint_list = solve_joint_distribution_iterate(p_a, p_b, min_value, max_value, num=num_joint)
                
    for joint in joint_list:
                    
        # Compute mean of fairness metrics
        eod = equal_opportunity_difference(joint, classifier_probs)
        
                        
        # Append results to data
        data.append({
                    'p_a': p_a,
                    'p_b': p_b,
                    'joint_distribution': joint,
                    'equal_opportunity': eod})
        
                
    data = pd.DataFrame(data)
                            
    # Flatten the 'p_a', 'p_b', and 'joint_distribution' columns
    flattened_p_a = flatten_dict_column(data, 'p_a')
    flattened_p_b = flatten_dict_column(data, 'p_b')
    flattened_joint = flatten_dict_column(data, 'joint_distribution')

    # Step 2: Concatenate the flattened columns with the original DataFrame
    df_final = pd.concat([data.drop(['p_a', 'p_b', 'joint_distribution'], axis=1), flattened_p_a, flattened_p_b, flattened_joint], axis=1)

    
    # Convert to DataFrame
    return df_final, true_eod


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


def generate_all_distributions_new_cond(joint, 
                               n_samples=100, classifier=DecisionTreeClassifier(), min_value = 0, max_value = 1, num_joint=50):
    """
    Generate all valid joint distributions, train models, and compute fairness metrics.
    
    Parameters:
        p_a_list (list): List of marginal probabilities for group A.
        p_b_list (list): List of marginal probabilities for group B.
        n_samples (int): Number of samples to draw for each distribution.
        classifier: ML model to use for fairness pipeline.
        num_joint (int): Number of joint distributions to generate.
    
    Returns:
        pd.DataFrame: DataFrame containing results for each valid distribution.
    """
     # 2. Compute marginals
    p_a, p_b = get_marginals(joint)
    
    p_b = break_x2_consistency(p_b, alpha=8, beta=1.0)
    
    print(p_a)
    print(p_b)
   
    # Dictionary to store all valid distributions for each (p_a, p_b)
    data = []
    
    # Sample training data and train model
    train_data_sample = sample_data_from_marginal(p_b, n_samples=n_samples)
    model = classifier_to_test(train_data_sample, classifier)
    classifier_probs = {
    x: model.predict_proba(pd.DataFrame([[x]], columns=['X2']))[0][1]
    for x in [0, 1]}
    
    print(classifier_probs)
    #classifier_probs = {x: model.predict_proba(np.array([[x]]))[0][1] for x in [0,1]}

    PAXY = {f'p{key[0]}{key[1]}{key[2]}': value for key, value in joint.items()}
    true_eod = equal_opportunity_difference(PAXY, classifier_probs)
    print(true_eod)

    # Generate joint distributions
    joint_list = solve_joint_distribution_iterate_cond(p_a, p_b, min_value, max_value, num=num_joint)
                    
    for joint in joint_list:
        
        print(joint)
                    
        # Compute mean of fairness metrics
        eod = equal_opportunity_difference(joint, classifier_probs)
        
        print(eod)
        
                        
        # Append results to data
        data.append({
                    'p_a': p_a,
                    'p_b': p_b,
                    'joint_distribution': joint,
                    'equal_opportunity': eod})
        
                
    data = pd.DataFrame(data)
                                
    # Flatten the 'p_a', 'p_b', and 'joint_distribution' columns
    flattened_p_a = flatten_dict_column(data, 'p_a')
    flattened_p_b = flatten_dict_column(data, 'p_b')
    flattened_joint = flatten_dict_column(data, 'joint_distribution')

    # Step 2: Concatenate the flattened columns with the original DataFrame
    df_final = pd.concat([data.drop(['p_a', 'p_b', 'joint_distribution'], axis=1), flattened_p_a, flattened_p_b, flattened_joint], axis=1)

    
    # Convert to DataFrame
    return df_final, true_eod