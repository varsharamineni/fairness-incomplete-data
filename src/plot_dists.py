import matplotlib.pyplot as plt

def plot_equal_opportunity_histograms(df_final, cols=10):
    """
    Plots histograms of 'mean_equal_opportunity' for each group defined by the columns starting with 'p_a' or 'p_b'.
    
    Parameters:
    df_final (pd.DataFrame): The input dataframe containing the probability columns starting with 'p_a' or 'p_b',
                              and 'mean_equal_opportunity' column.
    cols (int): Number of columns for the subplot layout (default is 10).
    """
    # Identify columns that start with "p_a" or "p_b"
    p_a_b_columns = [col for col in df_final.columns if col.startswith('p_a') or col.startswith('p_b')]

    # Group by the selected columns and aggregate 'mean_equal_opportunity' into a list
    grouped_df = df_final.groupby(p_a_b_columns)['equal_opportunity'].apply(list).reset_index()

    # Number of subplots (one for each group)
    num_groups = len(grouped_df)

    # Set up the subplots
    rows = (num_groups // cols) + (num_groups % cols > 0)  # Calculate number of rows required
    fig, axes = plt.subplots(rows, cols, figsize=(30, 5 * rows))
    axes = axes.flatten()  # Flatten the axes to make it easier to index

    # Loop through the groups and plot the histograms
    for idx, (row_idx, ax) in enumerate(zip(grouped_df.iterrows(), axes)):
        # Unpack group data
        _, row_data = row_idx
        group_label = f"p_a: {row_data['p_a_(0, 0)'], row_data['p_a_(1, 0)'], row_data['p_a_(0, 1)'], row_data['p_a_(1, 1)']} | " \
                      f"p_b: {row_data['p_b_(0, 0)'], row_data['p_b_(1, 0)'], row_data['p_b_(0, 1)'], row_data['p_b_(1, 1)']}"
        
        # Plot the histogram in the correct subplot
        ax.hist(row_data['equal_opportunity'], bins=10, edgecolor='black', color='skyblue')
        #ax.set_title(f"Histogram for Group: {group_label}")
        #ax.set_xlabel('Equal Opportunity')
        #ax.set_ylabel('Frequency')
        ax.grid(True)

    # Adjust layout for better spacing
    plt.tight_layout()
    plt.show()

# Example usage:
# plot_equal_opportunity_histograms(df_final)
