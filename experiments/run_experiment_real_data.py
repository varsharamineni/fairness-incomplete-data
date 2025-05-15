from joint_feasible_set.experiment_real_data import run_full_experiment_real_data
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.tree import DecisionTreeClassifier


pd.set_option('display.max_rows', None)   # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

alphas = [0.1, 0.3, 0.5, 0.7, 1.0] # Alpha values for consistency-breaking
betas =  [1.0, 3.0, 5.0, 7.0, 10.0]  # Beta values for consistency-breaking
num_candidates = 100     # Number of candidates to reconstruct

if __name__ == "__main__":
    # Set experiment parameters
    # Run the full experiment
    results_df = run_full_experiment_real_data(
        data_folder = 'real_data/processed/adult_clean.csv',
        s_var='capital-gain',
        o_var='marital-status',
        e_var='gender',
        alphas=alphas,
        betas=betas,
        num_candidates=num_candidates,
        seed=507
    )

    # Save the results to a CSV file
    results_df.to_csv("results/real_data/adult_data_di_dd_kl.csv", index=False)
    print("Experiment completed'.")
    print(results_df)  # Print the first few rows of results for verification
    
    
     # Filter the DataFrame for the desired alpha and beta values
    filtered_df = results_df[(results_df['alpha'] == 1.0) & (results_df['beta'] == 1.0)]
    print(filtered_df)

    # Flatten the plausible_metrics_di arrays into a single list
    plausible_metrics_di = np.concatenate(filtered_df['plausible_metrics_di'].values)
    true_di = filtered_df['true_di'].iloc[0]  # Assuming there's only one true value in the filtered DataFrame
    indep_given_overlap_di = filtered_df['indep_given_overlap_di'].iloc[0]
    marginal_preservation_di = filtered_df['marginal_preservation_di'].iloc[0]
    em_di = filtered_df['em_di'].iloc[0]


    
    # Plot the histogram
    sns.histplot(plausible_metrics_di, bins=20, color='steelblue')
    plt.axvline(x=true_di, color='red', linestyle='--', label='True')
    plt.axvline(x=indep_given_overlap_di, color='blue', linestyle='--', label='Indep Given Overlap')
    plt.axvline(x=marginal_preservation_di, color='green', linestyle='--', label='Marginal Preservation')
    plt.axvline(x=em_di, color='yellow', linestyle='--', label='Latent Bayes')
    plt.title("Distribution of Plausible Metrics (DI)")
    plt.xlabel("DI")
    plt.ylabel("Frequency")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.savefig('finalfigs/di_histogram_german_data.pdf', format='pdf')
    plt.show()
    
   