import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
FILE_S1 = 'results/plots/s1/s1_descriptive_stats.csv'
FILE_S2 = 'results/plots/s2_no_sum/s2_no_sum_descriptive_stats.csv'
OUTPUT_DIR = 'results/plots/comparison_results'

def main():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # 1. Load Data
    try:
        df_s1 = pd.read_csv(FILE_S1)
        df_s2 = pd.read_csv(FILE_S2)
    except FileNotFoundError:
        print("Error: Input CSV files not found.")
        return

    # 2. Tag Systems
    df_s1['System'] = 'S1_no_sum'
    df_s2['System'] = 'S2_no_sum'
    
    # Combine
    df_combined = pd.concat([df_s1, df_s2], ignore_index=True)
    
    # 3. Iterate through each Model type (Original, Spanish BT, Danish BT)
    unique_models = df_combined['Model'].unique()
    
    for model_name in unique_models:
        print(f"Generating plot for: {model_name}")
        
        # Filter for the current model and 'Race' grouping
        plot_data = df_combined[
            (df_combined['Model'] == model_name) & 
            (df_combined['Group Type'] == 'Race')
        ]

        if plot_data.empty:
            print(f"No data found for {model_name} with Race grouping.")
            continue

        # 4. Generate Plot
        plt.figure(figsize=(12, 7))
        sns.set_theme(style="whitegrid", context="talk")
        
        ax = sns.barplot(
            data=plot_data,
            x='Group',
            y='mean',
            hue='System',
            palette='viridis'
        )
        
        # Add labels
        plt.title(f'System Comparison: Accuracy by Racial Group ({model_name})', fontweight='bold')
        plt.ylabel('Mean Similarity Score')
        plt.xlabel('Identity Group')
        plt.ylim(0.85, 1.0) # Zoom in to see the difference clearly
        plt.legend(title='Chatbot System')

        # Add value labels on top of bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', padding=3, fontsize=10)

        # Save with a dynamic filename
        safe_model_name = model_name.replace(" ", "_").lower()
        save_path = os.path.join(OUTPUT_DIR, f'comparison_s1_s2_race_{safe_model_name}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to: {save_path}")
        plt.close() # Close figure to free memory

if __name__ == "__main__":
    main()