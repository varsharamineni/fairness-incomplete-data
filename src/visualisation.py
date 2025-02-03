import pandas as pd
import matplotlib.pyplot as plt

def analyze_equal_opportunity(df, show_plots=True, verbose=True):
    """
    Analyzes the distribution of equal opportunity values by grouping over p_a and p_b columns.
    
    Parameters:
        df (pd.DataFrame): The input DataFrame containing 'equal_opportunity' and 'p_a/p_b' columns.
        show_plots (bool): Whether to display histograms.
        verbose (bool): Whether to print summary statistics.
    
    Returns:
        pd.DataFrame: The grouped DataFrame with calculated statistics.
        dict: Summary statistics including average range, variance, and standard deviation.
    """
    # Select relevant columns
    p_a_b_columns = [col for col in df.columns if col.startswith('p_a') or col.startswith('p_b')]
    
    # Group by p_a and p_b columns
    grouped_df = df.groupby(p_a_b_columns)['equal_opportunity'].apply(list).reset_index()
    
    # Calculate min, max, and range of equal opportunity
    grouped_df['min_equal_opportunity'] = grouped_df['equal_opportunity'].apply(min)
    grouped_df['max_equal_opportunity'] = grouped_df['equal_opportunity'].apply(max)
    grouped_df['range_equal_opportunity'] = (
        grouped_df['max_equal_opportunity'] - grouped_df['min_equal_opportunity']
    )
    
    # Calculate variance and standard deviation
    grouped_df['variance_equal_opportunity'] = grouped_df['equal_opportunity'].apply(lambda x: pd.Series(x).var())
    grouped_df['std_equal_opportunity'] = grouped_df['equal_opportunity'].apply(lambda x: pd.Series(x).std())
    
    # Compute summary statistics
    stats = {
        "average_range": grouped_df['range_equal_opportunity'].mean(),
        "min_range": grouped_df['range_equal_opportunity'].min(),
        "max_range": grouped_df['range_equal_opportunity'].max(),
        "average_variance": grouped_df['variance_equal_opportunity'].mean(),
        "average_std": grouped_df['std_equal_opportunity'].mean()
    }
    
    # Print results if verbose is enabled
    if verbose:
        print(grouped_df)
        print("Average range of equal_opportunity:", stats["average_range"])
        print("Min range of equal_opportunity:", stats["min_range"])
        print("Max range of equal_opportunity:", stats["max_range"])
        print("Average Variance of equal_opportunity:", stats["average_variance"])
        print("Average Standard Deviation of equal_opportunity:", stats["average_std"])
    
    # Generate plots if show_plots is enabled
    if show_plots:
        plt.figure(figsize=(8, 5))
        plt.hist(grouped_df['range_equal_opportunity'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Ranges of Equal Opportunity', fontsize=14)
        plt.xlabel('Range of Equal Opportunity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
        
        plt.figure(figsize=(8, 5))
        plt.hist(grouped_df['variance_equal_opportunity'], bins=10, color='skyblue', edgecolor='black', alpha=0.7)
        plt.title('Distribution of Variance of Equal Opportunity', fontsize=14)
        plt.xlabel('Variance of Equal Opportunity', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.show()
    
    return grouped_df, stats


