import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_equal_opportunity_range_vs_joint_distribution(data):
    """
    Plots the range of equal opportunity grouped by columns starting with 'p_a' or 'p_b',
    and visualizes this range against joint distribution values, and also calculates the correlations 
    with a heatmap.

    Parameters:
    data (pd.DataFrame): The input dataframe containing 'equal_opportunity' column,
                          probability columns starting with 'p_a' or 'p_b',
                          and joint distribution columns starting with 'joint_distribution'.
    """
    # Identify columns that start with "p_a" or "p_b"
    probability_cols = [col for col in data.columns if col.startswith('p_a') or col.startswith('p_b')]
    
    # Group by the probability columns and compute the range of equal_opportunity
    grouped = data.groupby(probability_cols)['equal_opportunity'].agg(lambda x: x.max() - x.min()).reset_index()
    grouped.rename(columns={'equal_opportunity': 'equal_opportunity_range'}, inplace=True)
    
    # Identify joint distribution columns
    joint_cols = [col for col in data.columns if col.startswith('joint_distribution')]
    
    # Reset index of grouped to make it compatible
    grouped.reset_index(drop=True, inplace=True)
    
    # Reset index of joint_means and select the first row since it's just mean values
    joint_means = data[joint_cols].mean().to_frame().T.reset_index(drop=True)
    
    # Expand joint_means to match the length of grouped
    joint_means = pd.concat([joint_means] * len(grouped), ignore_index=True)
    
    # Concatenate along axis=1
    grouped = pd.concat([grouped, joint_means], axis=1)
    
    print(grouped.columns)
    
    # Plot the relationship between equal_opportunity_range and joint_distribution values
    plt.figure(figsize=(8, 5))
    
    for col in joint_cols:
        plt.scatter(grouped[col], grouped['equal_opportunity_range'], label=col)
    
    plt.xlabel("Joint Distribution Values")
    plt.ylabel("Range of Equal Opportunity")
    plt.title("Equal Opportunity Range vs Joint Distribution")
    plt.legend()
    plt.show()
    
    # Calculate correlation between 'equal_opportunity_range' and joint distribution columns
    correlations = grouped[joint_cols + ['equal_opportunity_range']].corr()
    print(correlations)
    
    # Plot heatmap of the correlations
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', fmt='.2f', vmin=-1, vmax=1, center=0)
    plt.title("Correlation Heatmap: Equal Opportunity Range vs Joint Distribution")
    plt.show()

# Example usage:
# plot_equal_opportunity_range_vs_joint_distribution(data)


    