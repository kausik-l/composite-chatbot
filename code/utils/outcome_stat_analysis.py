import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = 'data/outcome/s2_no_sum.csv'
OUTPUT_DIR = 'results/plots/s2_no_sum'

# Mapping codes to readable labels
RACE_LABELS = {0: 'Baseline', 1: 'African American', 2: 'European American'}
GENDER_LABELS = {0: 'Baseline', 1: 'Male', 2: 'Female'}

# Columns to analyze
OUTCOME_COLS = {
    'original_m_safechat_outcome': 'Original Model',
    'rt_es_m_safechat_outcome': 'Spanish Back-Trans',
    'rt_da_m_safechat_outcome': 'Danish Back-Trans'
}

def setup_plotting_style():
    """Sets a clean style for the plots."""
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams['figure.figsize'] = (12, 8)
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

def load_data(filepath):
    """Loads the dataset and maps readable labels."""
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found.")
        return None
    df = pd.read_csv(filepath)
    
    # Map labels for clearer grouping
    df['Race'] = df['Z_race'].map(RACE_LABELS)
    df['Gender'] = df['Z_gender'].map(GENDER_LABELS)
    
    return df

def calculate_descriptive_stats(df):
    """
    Calculates Mean and Standard Deviation for each model,
    grouped by Race and Gender.
    """
    print(f"\n{'='*20} DESCRIPTIVE STATISTICS {'='*20}")
    
    summary_list = []
    
    for col, model_name in OUTCOME_COLS.items():
        # 1. Group by Race
        race_stats = df.groupby('Race')[col].agg(['mean', 'std']).reset_index()
        race_stats['Model'] = model_name
        race_stats['Group Type'] = 'Race'
        race_stats.rename(columns={'Race': 'Group'}, inplace=True)
        summary_list.append(race_stats)
        
        # 2. Group by Gender
        gender_stats = df.groupby('Gender')[col].agg(['mean', 'std']).reset_index()
        gender_stats['Model'] = model_name
        gender_stats['Group Type'] = 'Gender'
        gender_stats.rename(columns={'Gender': 'Group'}, inplace=True)
        summary_list.append(gender_stats)
        
    full_summary = pd.concat(summary_list, ignore_index=True)
    
    # Reorder columns for readability
    full_summary = full_summary[['Model', 'Group Type', 'Group', 'mean', 'std']]
    
    print("\n--- Summary Table ---")
    print(full_summary)
    
    return full_summary

def plot_heatmaps(summary_df):
    """Generates heatmaps of the Mean Similarity Scores."""
    
    # --- Race Heatmap ---
    race_data = summary_df[summary_df['Group Type'] == 'Race']
    # Pivot: Index=Model, Columns=Group, Values=Mean
    pivot_race = race_data.pivot(index='Model', columns='Group', values='mean')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_race, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
    plt.title('Mean Similarity Scores by Race')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_scores_race.png'))
    plt.close()
    
    # --- Gender Heatmap ---
    gender_data = summary_df[summary_df['Group Type'] == 'Gender']
    pivot_gender = gender_data.pivot(index='Model', columns='Group', values='mean')
    
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_gender, annot=True, cmap="YlGnBu", fmt=".4f", linewidths=.5)
    plt.title('Mean Similarity Scores by Gender')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'heatmap_scores_gender.png'))
    plt.close()
    
    print("Saved heatmaps to output directory.")

def plot_score_distribution(df):
    """Violin plot showing the raw distribution of scores."""
    # Convert to long format for Seaborn
    df_long = df.melt(
        id_vars=['Race'], 
        value_vars=list(OUTCOME_COLS.keys()), 
        var_name='Model_Key', 
        value_name='Score'
    )
    df_long['Model'] = df_long['Model_Key'].map(OUTCOME_COLS)
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(
        data=df_long, 
        x='Model', 
        y='Score', 
        hue='Race', 
        split=False, 
        inner="quart",
        palette="muted"
    )
    
    plt.title('Distribution of Similarity Scores by Race')
    plt.ylim(0, 1.05) # Cosine similarity range
    plt.legend(title='Race Group', loc='lower right')
    plt.tight_layout()
    
    save_path = os.path.join(OUTPUT_DIR, 'dist_scores_race.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved distribution plot to {save_path}")

def main():
    setup_plotting_style()
    
    df = load_data(INPUT_FILE)
    if df is None:
        return
    
    # 1. Calculate Stats
    stats = calculate_descriptive_stats(df)
    
    # Save to CSV
    stats.to_csv(os.path.join(OUTPUT_DIR, 'descriptive_stats.csv'), index=False)
    
    # 2. Plot
    plot_heatmaps(stats)
    plot_score_distribution(df)
    
    print(f"\nDone! Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()