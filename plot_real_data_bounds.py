import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re



from paper_visualisations import *


pd.set_option('display.max_rows', None)   # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

if __name__ == "__main__":
    
    def str_to_float_list(s: str) -> list:

        # Replace np.float64(...) with just the number inside
        cleaned = re.sub(r'np\.float64\((.*?)\)', r'\1', s)
        
        # Convert cleaned string to a Python list using ast.literal_eval
        return ast.literal_eval(cleaned)
    
    
    results_df_adult = pd.read_csv("results/real_data/adult_data_di_dd_kl.csv")
    import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import ast
import re



from paper_visualisations import *


pd.set_option('display.max_rows', None)   # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns

if __name__ == "__main__":
    

    
    
    results_df_adult = pd.read_csv("results/real_data/adult_data_di_dd_kl.csv")
    results_df_compas = pd.read_csv("results/real_data/compas_data_di_dd_kl.csv")
    results_df_german = pd.read_csv("results/real_data/german_data_di_dd_kl.csv")
    
    
plot_fairness_metrics_from_df(results_df_adult, save_path="paperfigs/adult_hist.pdf")
plot_fairness_metrics_from_df(results_df_compas, save_path="paperfigs/compas_hist.pdf")
plot_fairness_metrics_from_df(results_df_german, save_path="paperfigs/german_hist.pdf")


for col in ['plausible_metrics_di', 'plausible_metrics_dd']:
    results_df_adult[col] = results_df[col].apply(str_to_float_list)
    
for col in ['plausible_metrics_di', 'plausible_metrics_dd']:
    results_df_compas[col] = results_df[col].apply(str_to_float_list)
        
results_df_compas = results_df_compas[(results_df_compas['alpha'] == 1.0) & (results_df_compas['beta'] == 1.0)]
results_df_adult = results_df_adult[(results_df['alpha'] == 1.0) & (results_df_adult['beta'] == 1.0)]
        

# Reference lines
true_di_adult = results_df_adult['true_di'].iloc[0]
marginal_preservation_di_adult = results_df_adult['marginal_preservation_di'].iloc[0]
em_di_adult = results_df_adult['em_di'].iloc[0]

true_di_compas = results_df_compas['true_di'].iloc[0]
marginal_preservation_di_compas = results_df_compas['marginal_preservation_di'].iloc[0]
em_di_compas = results_df_compas['em_di'].iloc[0]

# Plot
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# --- Adult ---
sns.histplot(all_di_adult, bins=10, color='lightgray', alpha=0.4, ax=axs[0])
axs[0].axvline(true_di_adult, color='red', linestyle='-', linewidth=2.5, label='True')
axs[0].axvline(marginal_preservation_di_adult, color='blue', linestyle='--', linewidth=2, label='Marginal')
axs[0].axvline(em_di_adult, color='green', linestyle='--', linewidth=2, label='Latent')
axs[0].set_title('Adult Dataset', fontsize=14)
axs[0].set_xlabel('Disparate Impact (DI)', fontsize=12)
axs[0].set_ylabel('Count', fontsize=12)
axs[0].grid(alpha=0.3)

# --- COMPAS ---
sns.histplot(all_di_compas, bins=10, color='lightgray', alpha=0.4, ax=axs[1])
axs[1].axvline(true_di_compas, color='red', linestyle='-', linewidth=2.5, label='True')
axs[1].axvline(marginal_preservation_di_compas, color='blue', linestyle='--', linewidth=2, label='Marginal')
axs[1].axvline(em_di_compas, color='green', linestyle='--', linewidth=2, label='Latent')
axs[1].set_title('COMPAS Dataset', fontsize=14)
axs[1].set_xlabel('Disparate Impact (DI)', fontsize=12)
axs[1].set_ylabel('Count', fontsize=12)
axs[1].grid(alpha=0.3)

# Combine legend from left plot
handles, labels = axs[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=3, frameon=False, fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
#plt.savefig(save_path, dpi=300, bbox_inches='tight', format='pdf')
plt.show()







    
    


    
    
    
    