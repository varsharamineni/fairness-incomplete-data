import numpy as np 
from sklearn.tree import DecisionTreeClassifier
from src.get_fairness_results import generate_all_distributions
from src.visualisation import analyze_equal_opportunity
from src.correlations import plot_equal_opportunity_range_vs_joint_distribution
from src.plot_dists import plot_equal_opportunity_histograms

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd



print("Script started")  # Debugging line

df = generate_all_distributions(p_a_range = np.arange(0.1, 0.9, 0.1), p_b_range = np.arange(0.1, 0.9, 0.1), 
                               n_samples=100, classifier=DecisionTreeClassifier(), min_value = 0.02, max_value = 0.4, num_joint=50)

#print(df)

analyze_equal_opportunity(df, show_plots=False, verbose=True)

#plot_equal_opportunity_histograms(df, cols=10)

#plot_equal_opportunity_range_vs_joint_distribution(df)


# Function to store results for different min_value and max_value
def store_equal_opportunity_results(min_values, max_values, p_a_range, p_b_range, n_samples, classifier):
    results = []
    
    for min_value in min_values:
        for max_value in max_values:
            # Generate all distributions (replace with your actual function)
            df = generate_all_distributions(p_a_range=p_a_range, p_b_range=p_b_range, 
                                            n_samples=n_samples, classifier=classifier,
                                            min_value=min_value, max_value=max_value, num_joint=50)

            # Analyze Equal Opportunity (replace with your actual function)
            grouped_df, eo_metrics = analyze_equal_opportunity(df, show_plots=False, verbose=False)
            
            # Extract the values from the stats dictionary
            results.append({
                'min_value': min_value,
                'max_value': max_value,
                'average_range': eo_metrics["average_range"],
                'min_range': eo_metrics["min_range"],
                'max_range': eo_metrics["max_range"],
                'average_variance': eo_metrics["average_variance"],
                'average_std': eo_metrics["average_std"]
            })
    
    # Convert the results list to a DataFrame
    results_df = pd.DataFrame(results)
    return results_df

# Define the range of min_value and max_value
min_values = np.arange(0.02, 0.1, 0.02)  # Example range
max_values = np.arange(0.2, 0.8, 0.1)  # Example range

# Define your parameters
p_a_range = np.arange(0.1, 0.9, 0.1)
p_b_range = np.arange(0.1, 0.9, 0.1)
n_samples = 100
classifier = DecisionTreeClassifier()

# Store the results
output_df = store_equal_opportunity_results(min_values, max_values, p_a_range, p_b_range, n_samples, classifier)

print(output_df)

# Plotting min_value against average_range
plt.figure(figsize=(8, 6))
plt.scatter(output_df['min_value'], output_df['average_range'], color='b', s=100, alpha=0.7)
plt.title('Min Threshold Value vs Average Range')
plt.xlabel('Min Threshold Value')
plt.ylabel('Average Range')
plt.grid(True)
plt.show()

# Plotting min_value against average_range
plt.figure(figsize=(8, 6))
plt.scatter(output_df['min_value'], output_df['max_range'], color='b', s=100, alpha=0.7)
plt.title('Min Threshold Value vs Max Range')
plt.xlabel('Min Threshold Value')
plt.ylabel('Max Range')
plt.grid(True)
plt.show()

# Plotting min_value against average_range
plt.figure(figsize=(8, 6))
plt.scatter(output_df['max_value'], output_df['average_range'], color='b', s=100, alpha=0.7)
plt.title('Max Threshold Value vs Average Range')
plt.xlabel('Max Threshold Value')
plt.ylabel('Average Range')
plt.grid(True)
plt.show()

# Plotting min_value against average_range
plt.figure(figsize=(8, 6))
plt.scatter(output_df['max_value'], output_df['max_range'], color='b', s=100, alpha=0.7)
plt.title('Max Threshold Value vs Max Range')
plt.xlabel('Max Threshold Value')
plt.ylabel('Max Range')
plt.grid(True)
plt.show()


# Create 3D figure
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot
sc = ax.scatter(output_df['min_value'], output_df['max_value'], output_df['average_range'], 
                c=output_df['average_range'], cmap='viridis', s=100, alpha=0.8)

# Labels and title
ax.set_xlabel('Min Value')
ax.set_ylabel('Max Value')
ax.set_zlabel('Average Range')
ax.set_title('3D Plot of Min Value, Max Value, and Average Range')

# Color bar
cbar = plt.colorbar(sc)
cbar.set_label('Average Range')

# Show plot
plt.show()



