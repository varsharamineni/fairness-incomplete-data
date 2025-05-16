import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import re


def plot_fairness_metrics_comparison(results_df, save_path="fairness_metrics_comparison.pdf"):

    sns.set_theme(style="whitegrid", palette="colorblind", font_scale=1.2)

    fig, axs = plt.subplots(2, 2, figsize=(14, 8))

    # Add a main heading
    
    axs[0,0].set_title("Histogram of Differences for True vs. Mean Fairness")
    axs[0,1].set_title("Scatter Plot for True vs. Mean Fairness")

    # Histogram Plot (top-left subplot)
    sns.histplot(results_df['true_mean_diff_di'], bins=20, color='steelblue', alpha = 0.4, edgecolor='black', ax=axs[0, 0])
    #axs[0, 0].set_title("Histogram of True DI − Mean DI Differences", fontsize=14)
    axs[0, 0].set_xlabel("True DI (from Ground Truth Joint) − Mean DI (Across Feasible Joint Distributions)", fontsize=12)
    axs[0, 0].set_ylabel("Frequency", fontsize=12)
    axs[0, 0].legend(fontsize=10)
    axs[0, 0].grid(True, linestyle='--', alpha=0.5)


    # Scatter Plot (top-right subplot)
    sns.scatterplot(data=results_df, x='true_di', y='mean_di', s=50, color='steelblue', edgecolor='black', ax=axs[0, 1])
    axs[0, 1].plot([0, 5], [0, 5], '--', color='red', linewidth=0.7, label='Perfect Agreement (y = x)')
    #axs[0, 1].set_title("Scatter Plot: True DI vs. Mean DI", fontsize=14)
    axs[0, 1].set_xlabel("True DI (from Ground Truth Joint)", fontsize=12)
    axs[0, 1].set_ylabel("Mean DI (Across Feasible Joint Distributions)", fontsize=12)
    axs[0, 1].grid(True, linestyle='--', alpha=0.5)

    # Histogram Plot (bottom-left subplot)
    sns.histplot(results_df['true_mean_diff_dd'], bins=20, color='#1b7764', alpha = 0.4, edgecolor='black', ax=axs[1, 0])
    #axs[1, 0].set_title("Histogram of True DD − Mean DD Differences", fontsize=14)
    axs[1, 0].set_xlabel("True DD (from Ground Truth Joint) − Mean DD (Across Feasible Joint Distributions)", fontsize=12)
    axs[1, 0].set_ylabel("Frequency", fontsize=12)
    axs[1, 0].grid(True, linestyle='--', alpha=0.5)


    # Scatter Plot (bottom-right subplot)
    sns.scatterplot(data=results_df, x='true_dd', y='mean_dd', s=50, color='#1b7764', edgecolor='black', ax=axs[1, 1])
    axs[1, 1].plot([-1, 1], [-1, 1], '--', color='red', linewidth=0.7, label='Perfect Agreement (y = x)')
    #axs[1, 1].set_title("Scatter Plot: True DD vs. Mean DD", fontsize=14)
    axs[1, 1].set_xlabel("True DD (from Ground Truth Joint)", fontsize=12)
    axs[1, 1].set_ylabel("Mean DD (Across Feasible Joint Distributions)", fontsize=12)
    axs[1, 1].grid(True, linestyle='--', alpha=0.5)

    # Adjust layout and save
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
    plt.show()
    
    
def str_to_float_list(s: str) -> list:

        # Replace np.float64(...) with just the number inside
    cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', s)
        
        # Convert cleaned string to a Python list using ast.literal_eval
    return ast.literal_eval(cleaned)
    
def plot_fairness_ranges(results_df, save_path="paperfigs/fairness_ranges.pdf"):
        """
        Plot and save side-by-side boxplots for range_dd and range_di.

        Args:
            results_df (pd.DataFrame): DataFrame with columns 'range_dd' and 'range_di'.
            save_folder (str): Path to the folder where the plot will be saved.
            filename (str): Name of the output file (default: 'fairness_ranges.png').
        """
        sns.set_theme(style="whitegrid", font_scale=1.2)

        # Initialize figure with two subplots
        fig, axs = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
        
        # Disparate Impact (DI)
        boxplot_di = sns.boxplot(
            y=results_df['range_di'], 
            color='lightblue', 
            width=0.4, 
            showfliers=False, 
            ax=axs[0]        )

        axs[0].set_title("Disparate Impact (DI)")
        axs[0].set_ylabel("")  # Avoid repeating y-label
        axs[0].grid(axis='y', linestyle='--', alpha=0.7)

        # Demographic Disparity (DD)
        boxplot_dd = sns.boxplot(
            y=results_df['range_dd'], 
            color='#2ab79a',
            width=0.4, 
            showfliers=False, 
            ax=axs[1]        )
        for patch in boxplot_dd.artists:
            patch.set_facecolor('#1b7764')
            patch.set_alpha(0.6)  # Set transparency
        
        axs[1].set_title("Demographic Disparity (DD)")
        axs[1].set_ylabel("Range")
        axs[1].grid(axis='y', linestyle='--', alpha=0.7)

        # Main title
        fig.suptitle("Ranges for Possible Fairness Metrics across Feasible Joint Distributions", fontsize=14)

        plt.tight_layout(rect=[0, 0.03, 1, 0.99])  # Leave space for suptitle

        # Save and show
        plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
        plt.show()
        
def plot_fairness_metrics_from_df(results_df, save_path):
        
    for col in ['plausible_metrics_di', 'plausible_metrics_dd']:
        results_df[col] = results_df[col].apply(str_to_float_list)
        
    results_df = results_df[(results_df['alpha'] == 1.0) & (results_df['beta'] == 1.0)]
        
        # Flatten arrays from columns with lists/arrays of metrics
    all_di = np.concatenate(results_df['plausible_metrics_di'].values)
    all_dd = np.concatenate(results_df['plausible_metrics_dd'].values)

        # Extract reference lines (take first row values)
    true_di = results_df['true_di'].iloc[0]
    marginal_preservation_di = results_df['marginal_preservation_di'].iloc[0]
    em_di = results_df['em_di'].iloc[0]

    true_dd = results_df['true_dd'].iloc[0]
    marginal_preservation_dd = results_df['marginal_preservation_dd'].iloc[0]
    em_dd = results_df['em_dd'].iloc[0]

    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # Plot DI histogram + lines
    sns.histplot(all_di, bins=10, color='lightgray', alpha=0.3, ax=axs[0])
    axs[0].axvline(true_di, color='red', linestyle='-', linewidth=2.5, label='True')
    axs[0].axvline(marginal_preservation_di, color='blue', linestyle='--', linewidth=2, label='Marginal Preservation')
    axs[0].axvline(em_di, color='green', linestyle='--', linewidth=2, label='Latent Naïve Bayes')
    axs[0].set_title('Histogram of Possible DI Metrics')
    axs[0].set_xlabel('Disparate Impact (DI)', fontsize=12)
    axs[0].set_ylabel('Count', fontsize=12)
    #axs[0].legend()
    axs[0].grid(alpha=0.3)

        # Plot DD histogram + lines
    sns.histplot(all_dd, bins=10, color='lightgray', alpha=0.3, ax=axs[1])
    axs[1].axvline(true_dd, color='red', linestyle='-', linewidth=2.5, label='True')
    axs[1].axvline(marginal_preservation_dd, color='blue', linestyle='--', linewidth=2, label='Marginal Preservation')
    axs[1].axvline(em_dd, color='green', linestyle='--', linewidth=2, label='Latent Naïve Bayes')
    axs[1].set_title('Histogram of Possible DD Metrics')
    axs[1].set_xlabel('Demographic Disparity (DD)', fontsize=12)
    axs[1].set_ylabel('Count', fontsize=12)
    #axs[1].legend('')
    axs[1].grid(alpha=0.3)
        
            # Create combined legend from first subplot handles and labels
    handles, labels = axs[0].get_legend_handles_labels()

        # Add legend to figure
    fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=12)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space at top for legend
    plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')

    plt.show()
        
        
        
