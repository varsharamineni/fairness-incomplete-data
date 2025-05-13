import pandas as pd
import numpy as np

from inconst_joint.experiment import run_full_experiment
from inconst_joint.visualise import *
from paper_visualisations import *


pd.set_option('display.max_rows', None)   # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

if __name__ == "__main__":
    
    results_df = pd.read_csv("results/experiment_results_1000_di_dd_kl.csv")
    
    bins = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 10.0]
    labels = ['0–0.05', '0.05–0.1', '0.1–0.2', '0.2–0.3', '0.3–0.5', '0.5+']
    results_df['kl_bin'] = pd.cut(results_df['const'], bins=bins, labels=labels, include_lowest=True)
    
    results_df = results_df[results_df['true_di'] <= 5.0]
    
    #plot_fairness_metrics_comparison(results_df, save_path="paperfigs/fairness_metrics_comparison.pdf")
    





    
    
    
    
    plt.figure(figsize=(6, 5))
    sns.boxplot(y=results_df['true_mean_diff_dd'], color='lightblue', hue=results_df['kl_bin'], width=0.3, showfliers=False)

    plt.title("Distribution of True - Mean of Plausible Fairness (Disparate Impact)")
    plt.ylabel("")
    plt.xlabel("")  # Leave blank or put a label like "Fairness Metric" if you prefer
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    #plt.savefig('finalfigs/range_boxplot.pdf', format='pdf')
    plt.show()
    
    
   
   
        # Optional: Improve plot style
    sns.set(style="whitegrid")

    # Scatter plot: KL divergence vs absolute error
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=results_df['const'],
        y=abs(results_df['true_mean_diff_di']),
        alpha=0.6,
        edgecolor='k'
    )

    # Optional threshold line for visual reference
    plt.axhline(0.05, color='red', linestyle='--', label='Error = 0.05')

    plt.xlabel("KL Divergence Between Marginals")
    plt.ylabel("Absolute Error (Mean DI - True DI)")
    plt.title("Robustness of Average Fairness Metric Under Marginal Inconsistency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    
    
          # Optional: Improve plot style
    sns.set(style="whitegrid")

    # Scatter plot: KL divergence vs absolute error
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        x=results_df['const'],
        y=abs(results_df['avg_kl_y_given_e']),
        alpha=0.6,
        edgecolor='k'
    )

    # Optional threshold line for visual reference
    plt.axhline(0.05, color='red', linestyle='--', label='Error = 0.05')

    plt.xlabel("KL Divergence Between Marginals")
    plt.ylabel("KL Divergence Between True and Estimated p(y=1 | e)")
    #plt.title("Robustness of Average Fairness Metric Under Marginal Inconsistency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
   
   
    print(results_df.describe())
    
    print(results_df.groupby('kl_bin').describe())

    
    

