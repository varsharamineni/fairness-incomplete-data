import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_histograms(results_df, metric='range_dd'):
    """
    Plot histogram for a given metric across all trials.

    Args:
        results_df (pd.DataFrame): DataFrame of experiment results.
        metric (str): Column name to plot.
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(results_df[metric], bins=20, kde=True)
    plt.title(f"Histogram of {metric}")
    plt.xlabel(metric)
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

    
def plot_grouped_histograms(data, col="alpha", row="beta", metric='range_dd'):
    
    # Group the results by alpha and beta
    g = sns.FacetGrid(data, col=col, row=row, margin_titles=True)
    g.map(sns.histplot, metric, kde=True, bins=20, color="blue")

    g.set_axis_labels(f' {metric}', 'Density')
    g.set_titles(col_template='Alpha={col_name}', row_template='Beta={row_name}')
    plt.show()


def plot_true_vs_mean(results_df, metric_name='di'):
    """
    Scatter plot of true vs mean reconstructed metric, with separate lines per alpha-beta group.

    Args:
        results_df (pd.DataFrame): DataFrame with true and mean metric columns.
        metric_name (str): 'di' for disparate impact or 'dd' for demographic disparity.
    """
    true_col = f"true_{metric_name}"
    mean_col = f"mean_{metric_name}"
    
    # Combine alpha and beta into a string for hue/grouping
    results_df['alpha_beta'] = results_df['alpha'].astype(str) + '-' + results_df['beta'].astype(str)

    plt.figure(figsize=(7, 7))

    # Scatter plot
    sns.scatterplot(
        data=results_df, 
        x=true_col, 
        y=mean_col, 
        hue='alpha_beta', 
        s=60
    )

    # Reference line y = x
    plt.plot([0, 1], [0, 1], '--', color='gray')

    plt.xlabel(f"True {metric_name.upper()}")
    plt.ylabel(f"Mean Reconstructed {metric_name.upper()}")
    plt.title(f"True vs. Mean {metric_name.upper()} Across Runs")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_grouped_true_vs_mean(data, col="alpha", row="beta", metric="dd"):
    """
    Plots true vs. mean metric (e.g., DD or DI) grouped by alpha and beta.
    """
    x_col = f"true_{metric}"
    y_col = f"mean_{metric}"

    g = sns.FacetGrid(data, col=col, row=row, margin_titles=True, height=4)
    g.map_dataframe(sns.scatterplot, x=x_col, y=y_col, alpha=0.7)
    g.map_dataframe(sns.lineplot, x=x_col, y=x_col, color="red", linestyle="--")  # y=x reference line

    g.set_axis_labels(f'True {metric.upper()}', f'Mean {metric.upper()}')
    g.set_titles(col_template='Alpha={col_name}', row_template='Beta={row_name}')
    plt.subplots_adjust(top=0.9)
    g.fig.suptitle(f'True vs Mean {metric.upper()} Across Alpha-Beta Grid')
    plt.show()
    
    


def plot_metric_range_check(results_df, metric_prefix='di', index_col='trial'):
    """
    Plots true metric value with min/max reconstructed bounds and highlights out-of-range points.

    Args:
        results_df (pd.DataFrame): The results DataFrame.
        metric_prefix (str): One of 'eod', 'di', 'dd', etc.
        index_col (str): Column to use for x-axis (e.g., 'trial').
    """
    true_col = f'true_{metric_prefix}'
    min_col = f'min_{metric_prefix}'
    max_col = f'max_{metric_prefix}'

    # Copy dataframe and compute range check
    results_df = results_df.copy()
    results_df['is_within_range'] = (results_df[true_col] >= results_df[min_col]) & (results_df[true_col] <= results_df[max_col])

    # Percentage of points within range
    percentage_within_range = results_df['is_within_range'].mean() * 100

    # Create figure and scatter plot
    plt.figure(figsize=(12, 7))
    x_vals = results_df[index_col]

    # Plot true metric values (smaller dots for better clarity)
    plt.scatter(x_vals, results_df[true_col], color='blue', label=f'True {metric_prefix.upper()}', s=40, alpha=0.7, zorder=5)

    # Plot min and max bounds as dashed lines
    plt.plot(x_vals, results_df[min_col], 'g--', label=f'Min {metric_prefix.upper()}', zorder=3)
    plt.plot(x_vals, results_df[max_col], 'r--', label=f'Max {metric_prefix.upper()}', zorder=3)

    # Highlight out-of-range values (red dots, smaller size)
    out_of_range = ~results_df['is_within_range']
    plt.scatter(x_vals[out_of_range], results_df[true_col][out_of_range], color='red', label='Out of Range', s=30, alpha=0.9, zorder=6)

    # Annotate percentage of values within bounds
    plt.text(0.5, 0.9, f'Within bounds: {percentage_within_range:.2f}%', fontsize=12,
             ha='center', va='center', transform=plt.gca().transAxes, color='green')

    # Titles and labels with improved clarity
    plt.xlabel('Trial Index', fontsize=14)
    plt.ylabel(f'{metric_prefix.upper()} Value', fontsize=14)
    plt.title(f'True {metric_prefix.upper()} vs. Reconstructed Range', fontsize=16)
    plt.legend(title='Metric Bounds', fontsize=12, loc='upper right')
    
    # Grid and layout
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()




