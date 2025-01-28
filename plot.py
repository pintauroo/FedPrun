import os
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style for seaborn
sns.set(style="whitegrid")

# Define the paths to directories
RESULTS_DIR = 'results'
CONVERGENCE_PLOTS_DIR = 'convergence_plots'  # Not used in this script but kept for consistency
BAR_PLOTS_DIR = 'bar_plots'

# Create directories to save the plots
os.makedirs(CONVERGENCE_PLOTS_DIR, exist_ok=True)
os.makedirs(BAR_PLOTS_DIR, exist_ok=True)

# Initialize a list to hold all the data
data = []

# Regular expression to extract the pruning value from the filename
# Assumes filenames are in the format 'iid_<value>.json'
filename_pattern = re.compile(r'niid_(\d+)\.json')

# Iterate over all files in the results directory
for filename in os.listdir(RESULTS_DIR):
    match = filename_pattern.match(filename)
    if match:
        pruning_value = float(match.group(1))  # Extracted as integer, convert to float
        file_path = os.path.join(RESULTS_DIR, filename)
        
        # Load JSON data
        with open(file_path, 'r') as f:
            try:
                json_data = json.load(f)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON from file {filename}: {e}")
                continue
        
        # Iterate over each pruning policy in the JSON data
        for entry in json_data:
            pruning_policy = entry.get("pruning_policy", "Unknown")
            pruning_amount = entry.get("pruning_amount", 0.0)
            model_size_before = entry.get("model_size_before_pruning_MB", None)
            model_size_after = entry.get("model_size_after_pruning_MB", None)
            rounds = entry.get("rounds", [])
            
            # Iterate over each round
            for round_info in rounds:
                round_number = round_info.get("round", None)
                avg_train_loss = round_info.get("avg_train_loss", None)
                test_loss = round_info.get("test_loss", None)
                test_accuracy = round_info.get("test_accuracy", None)
                model_size_mb = round_info.get("model_size_MB", None)
                
                # Append the data to the list
                data.append({
                    'Pruning Value': pruning_value,
                    'Pruning Policy': pruning_policy,
                    'Pruning Amount': pruning_amount,
                    'Round': round_number,
                    'Avg Train Loss': avg_train_loss,
                    'Test Loss': test_loss,
                    'Test Accuracy': test_accuracy,
                    'Model Size (MB)': model_size_mb
                })
    else:
        print(f"Filename {filename} does not match the pattern and will be skipped.")

# Create a DataFrame from the data
df = pd.DataFrame(data)

# Display the DataFrame (optional)
print("Per-Round Data Sample:")
print(df.head())

# Ensure correct data types
df['Pruning Value'] = df['Pruning Value'].astype(float)
df['Round'] = df['Round'].astype(int)
df['Pruning Policy'] = df['Pruning Policy'].astype(str)

# Sort the DataFrame for better processing
df = df.sort_values(['Pruning Policy', 'Pruning Value', 'Round'])

# =======================
# Compute Final Metrics
# =======================

# Compute Final Test Loss and Final Test Accuracy from the last round
final_metrics_df = df.groupby(['Pruning Policy', 'Pruning Value']).apply(
    lambda x: x.sort_values('Round').iloc[-1]
).reset_index(drop=True)[['Pruning Policy', 'Pruning Value', 'Test Loss', 'Test Accuracy', 'Model Size (MB)']]

# Rename columns for clarity
final_metrics_df.rename(columns={
    'Test Loss': 'Final Test Loss',
    'Test Accuracy': 'Final Test Accuracy',
    'Model Size (MB)': 'Model Size After Pruning (MB)'
}, inplace=True)

# Extract Model Size Before Pruning from the first round
model_size_before_df = df.groupby(['Pruning Policy', 'Pruning Value']).apply(
    lambda x: x.sort_values('Round').iloc[0]
).reset_index(drop=True)[['Pruning Policy', 'Pruning Value', 'Model Size (MB)']]

model_size_before_df.rename(columns={'Model Size (MB)': 'Model Size Before Pruning (MB)'}, inplace=True)

# Merge to get a complete final_metrics_df
final_metrics_df = pd.merge(final_metrics_df, model_size_before_df, on=['Pruning Policy', 'Pruning Value'])

# Calculate Model Size Reduction
final_metrics_df['Model Size Reduction (MB)'] = final_metrics_df['Model Size Before Pruning (MB)'] - final_metrics_df['Model Size After Pruning (MB)']

# Reorder columns for better readability
final_metrics_df = final_metrics_df[[
    'Pruning Policy', 'Pruning Value',
    'Model Size Before Pruning (MB)', 'Model Size After Pruning (MB)',
    'Model Size Reduction (MB)', 'Final Test Loss', 'Final Test Accuracy'
]]

print("\nFinal Metrics DataFrame:")
print(final_metrics_df)

# =======================
# Bar Plots for Final Test Accuracy and Final Test Loss
# =======================

# Define bar plot metrics
bar_metrics = {
    'Final Test Accuracy': {
        'ylabel': 'Final Test Accuracy',
        'title': 'Final Test Accuracy by Pruning Value and Policy',
        'palette': 'viridis'
    },
    'Final Test Loss': {
        'ylabel': 'Final Test Loss',
        'title': 'Final Test Loss by Pruning Value and Policy',
        'palette': 'magma'
    }
}

# Function to create grouped bar plots
def create_grouped_bar_plot(df, metric, props):
    plt.figure(figsize=(14, 8))
    
    # Create a new column for combined Pruning Policy and Value for better grouping
    df['Policy & Value'] = df['Pruning Policy'] + f" (Pruning Value: {df['Pruning Value']})"
    
    # Sort the policies for consistent color mapping
    df = df.sort_values('Policy & Value')
    
    # Create barplot
    sns.barplot(
        data=df,
        x='Pruning Value',
        y=metric,
        hue='Pruning Policy',
        palette=props['palette'],
        ci=None
    )
    
    plt.title(props['title'], fontsize=16)
    plt.xlabel('Pruning Value', fontsize=14)
    plt.ylabel(props['ylabel'], fontsize=14)
    
    # Add value labels on top of each bar
    for container in plt.gca().containers:
        plt.gca().bar_label(container, fmt='%.2f', label_type='edge', fontsize=10)
    
    plt.legend(title='Pruning Policy', fontsize=12, title_fontsize=13, loc='best')
    plt.tight_layout()
    
    # Save the plot to the bar_plots directory
    metric_filename = metric.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    plot_filename = os.path.join(BAR_PLOTS_DIR, f"{metric_filename}.png")
    plt.savefig(plot_filename, dpi=300)
    plt.close()
    
    print(f"Saved bar plot for {metric} as {plot_filename}")

# Generate bar plots for Final Test Accuracy and Final Test Loss
for metric, props in bar_metrics.items():
    create_grouped_bar_plot(final_metrics_df, metric, props)

# =======================
# Optional: Displaying Plots Inline (For Jupyter Notebooks)
# =======================

# If you are using a Jupyter notebook and want to display the plots inline,
# you can uncomment the following lines:

# from IPython.display import display, Image
# # Display Bar Plots
# for metric in bar_metrics.keys():
#     plot_filename = os.path.join(BAR_PLOTS_DIR, f"{metric.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')}.png")
#     display(Image(filename=plot_filename))
