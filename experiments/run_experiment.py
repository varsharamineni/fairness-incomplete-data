from inconst_joint.experiment import run_full_experiment, run_systematic_experiment
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    # Set experiment parameters
    num_trials = 1000
    alphas = [0.1, 0.3, 0.5, 0.7, 1.0] # Alpha values for consistency-breaking
    betas =  [1.0, 3.0, 5.0, 7.0, 10.0]  # Beta values for consistency-breaking
    num_candidates = 100     # Number of candidates to reconstruct

    # Run the full experiment
    results_df = run_full_experiment(
        num_trials=num_trials,
        alphas=alphas,
        betas=betas,
        num_candidates=num_candidates
    )

    # Save the results to a CSV file
    results_df.to_csv("experiment_results_1000_di_dd_kl.csv", index=False)
    print("Experiment completed. Results saved to 'experiment_results.csv'.")
    print(results_df.head())  # Print the first few rows of results for verification
  

    
    
    
    
    