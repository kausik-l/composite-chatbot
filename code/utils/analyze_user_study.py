import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Configuration
INPUT_FILE = "data/user_study.csv"
OUTPUT_DIR = "data/user_study_results"

# -----------------------------
# Consistent styling everywhere
# -----------------------------
WORKFLOW_ORDER = ["W1", "W2", "W3"]

# Use fixed HEX codes (Matplotlib tab colors) for identical appearance everywhere
WORKFLOW_COLORS = {
    "W1": "#2ca02c",  # tab:green
    "W2": "#d62728",  # tab:red
    "W3": "#ff7f0e",  # tab:orange
}

def enforce_workflow_order(x):
    return pd.Categorical(x, categories=WORKFLOW_ORDER, ordered=True)



def load_and_prep_data():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    df = pd.read_csv(INPUT_FILE)
    
    # 1. Precise Column Mapping
    # 0: Timestamp
    # 1: Exp_AI ("On a scale of 1–5, how familiar are you with the development of AI systems?")
    # 2: Exp_Fairness
    # 3-5: W1 Correctness (Q1, Q2, Overall)
    # 6-8: W2 Correctness (Q1, Q2, Overall)
    # 9-11: W3 Correctness (Q1, Q2, Overall)
    # 12: Auto Agreement (Correctness Panel)
    # 13: Confidence (Correctness)
    # 14: W1 Fairness
    # 15: W2 Fairness
    # 16: W3 Fairness
    # 17: Auto Agreement (Fairness Panel)
    # 18: Confidence (Fairness)
    # 19: Choice Efficiency Ex 1 ("Which workflow is best...")
    # 20: Choice Efficiency Ex 2 ("Which workflow is best...")
    # 21: Auto Agreement (Efficiency Panel)
    # 22: Confidence (Efficiency)
    
    new_cols = [
        "Timestamp", "Exp_AI", "Exp_Fairness",
        "W1_Corr_1", "W1_Corr_2", "W1_Corr_Overall",
        "W2_Corr_1", "W2_Corr_2", "W2_Corr_Overall",
        "W3_Corr_1", "W3_Corr_2", "W3_Corr_Overall",
        "Auto_Agree_Corr", "Conf_Corr",
        "W1_Fair_Overall", "W2_Fair_Overall", "W3_Fair_Overall",
        "Auto_Agree_Fair", "Conf_Fair",
        "Choice_Effic_1", "Choice_Effic_2",
        "Auto_Agree_Effic", "Conf_Effic"
    ]
    
    # Handle length mismatch
    current_cols = df.columns.tolist()
    if len(current_cols) > len(new_cols):
        extra = [f"Extra_{i}" for i in range(len(current_cols) - len(new_cols))]
        new_cols.extend(extra)
    elif len(current_cols) < len(new_cols):
        new_cols = new_cols[:len(current_cols)]
        
    df.columns = new_cols
    
    # 2. Extract Experience Ratings
    def extract_rating(val):
        val_str = str(val).lower()
        if "no prior" in val_str: return 1
        if "little experience" in val_str: return 2
        if "some experience" in val_str: return 3
        if "moderate experience" in val_str: return 4
        if "extensive experience" in val_str: return 5
        # Fallback to checking for digits if text doesn't match
        for char in val_str:
            if char.isdigit(): return int(char)
        # return 1 
            
    df['Exp_AI_Score'] = df['Exp_AI'].apply(extract_rating)
    df['Exp_Fair_Score'] = df['Exp_Fairness'].apply(extract_rating)
    
    # 3. Numeric Conversions
    numeric_cols = [c for c in df.columns if "W" in c or "Auto" in c or "Conf" in c]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # 4. Compute Average Correctness Scores (Q1 + Q2) / 2
    # Instead of relying on "Overall" column, we average the individual question ratings
    df['W1_Avg_Corr'] = (df['W1_Corr_1'] + df['W1_Corr_2']) / 2.0
    df['W2_Avg_Corr'] = (df['W2_Corr_1'] + df['W2_Corr_2']) / 2.0
    df['W3_Avg_Corr'] = (df['W3_Corr_1'] + df['W3_Corr_2']) / 2.0

    return df

# ---------------------------------------------------------
# RH1: Correctness (ALL USERS)
# ---------------------------------------------------------
def analyze_rh1_correctness(df):
    print("\n" + "="*60)
    print("RH1: Correctness Preference (All Users)")
    print("="*60)
    
    # Use computed averages instead of overall rating column
    metrics = ['W1_Avg_Corr', 'W2_Avg_Corr', 'W3_Avg_Corr']
    print(df[metrics].describe().loc[['mean', 'std']])
    
    # try:
    stat, p = stats.friedmanchisquare(df[metrics[0]], df[metrics[1]], df[metrics[2]])
    print(f"Friedman Test: Statistic={stat:.3f}, p-value={p:.4f}")
    if p < 0.05:
        print("  -> Significant. Pairwise Wilcoxon (Bonferroni):")
        pairs = [('W1', 'W2'), ('W1', 'W3'), ('W2', 'W3')]
        col_map = {'W1': metrics[0], 'W2': metrics[1], 'W3': metrics[2]}
        for a, b in pairs:
            s, p_w = stats.wilcoxon(df[col_map[a]], df[col_map[b]])
            print(f"    {a} vs {b}: p_adj={min(p_w*3, 1.0):.4f}")
    # except ValueError: pass

    # plt.figure(figsize=(8, 6))
    # df_melt = df.melt(value_vars=metrics, var_name='Workflow', value_name='Score')
    # # Update map to reflect calculated average
    # df_melt['Workflow'] = df_melt['Workflow'].map({'W1_Avg_Corr':'W1', 'W2_Avg_Corr':'W2', 'W3_Avg_Corr':'W3'})
    
    # sns.boxplot(
    #     x='Workflow', y='Score', data=df_melt, palette="Set2",
    #     showmeans=True, meanline=True, 
    #     meanprops={"color": "red", "linewidth": 1.5, "linestyle": "--"}
    # )
    # plt.title("RH1: Workflow Correctness (Avg of Q1 & Q2)")
    # plt.ylabel("Likert Score (1-5)")
    # plt.ylim(1, 5.5)
    # plt.tight_layout()
    # plt.savefig(os.path.join(OUTPUT_DIR, "rh1_correctness_all.png"))
    plt.figure(figsize=(8, 6))
    df_melt = df.melt(value_vars=metrics, var_name='Workflow', value_name='Score')
    df_melt['Workflow'] = df_melt['Workflow'].map({
        'W1_Avg_Corr':'W1', 'W2_Avg_Corr':'W2', 'W3_Avg_Corr':'W3'
    })
    df_melt['Workflow'] = enforce_workflow_order(df_melt['Workflow'])

    sns.boxplot(
    x="Workflow", y="Score", data=df_melt,
    order=WORKFLOW_ORDER,
    palette=WORKFLOW_COLORS,
    saturation=1,              # <-- key: prevents seaborn from muting colors
    showmeans=True, meanline=True,
    meanprops={"color": "black", "linewidth": 1.5, "linestyle": "--"}
    )

    plt.title("RH1: Workflow Correctness (Avg of Q1 & Q2)")
    plt.ylabel("Likert Score (1-5)")
    plt.ylim(1, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "rh1_correctness_all.png"))

    print("Saved plot: rh1_correctness_all.png")

# ---------------------------------------------------------
# RH2: Fairness (SPLIT by Fairness Awareness)
# ---------------------------------------------------------
def analyze_rh2_fairness(df):
    print("\n" + "="*60)
    print("RH2: Fairness Preference (Split by Fairness Awareness)")
    print("="*60)
    
    df_unaware = df[df['Exp_Fair_Score'] == 1]
    df_aware = df[df['Exp_Fair_Score'] >= 2]
    
    metrics = ['W1_Fair_Overall', 'W2_Fair_Overall', 'W3_Fair_Overall']
    groups = [("Unaware (Score=1)", df_unaware), ("Aware (Score>=2)", df_aware)]
    
    for label, sub_df in groups:
        print(f"\n--- Group: {label} (N={len(sub_df)}) ---")
        if len(sub_df) < 2: 
            print("N too small.")
            continue
            
        print(sub_df[metrics].describe().loc[['mean', 'std']])
        
        try:
            stat, p = stats.friedmanchisquare(sub_df[metrics[0]], sub_df[metrics[1]], sub_df[metrics[2]])
            print(f"Friedman Test: Statistic={stat:.3f}, p-value={p:.4f}")
            if p < 0.05:
                print("  -> Significant. Pairwise Wilcoxon (Bonferroni):")
                pairs = [('W1', 'W2'), ('W1', 'W3'), ('W2', 'W3')]
                col_map = {'W1': metrics[0], 'W2': metrics[1], 'W3': metrics[2]}
                for a, b in pairs:
                    s, p_w = stats.wilcoxon(df[col_map[a]], df[col_map[b]])
                    print(f"    {a} vs {b}: p_adj={min(p_w*3, 1.0):.4f}")
        except: pass
        
        # plt.figure(figsize=(8, 6))
        # df_melt = sub_df.melt(value_vars=metrics, var_name='Workflow', value_name='Score')
        # df_melt['Workflow'] = df_melt['Workflow'].map({'W1_Fair_Overall':'W1', 'W2_Fair_Overall':'W2', 'W3_Fair_Overall':'W3'})
        
        # sns.boxplot(
        #     x='Workflow', y='Score', data=df_melt, palette="Set3",
        #     showmeans=True, meanline=True, 
        #     meanprops={"color": "blue", "linewidth": 1.5, "linestyle": "--"}
        # )
        # plt.title(f"RH2: Fairness ({label})")
        # plt.ylim(1, 5.5)
        # plt.tight_layout()
        # safe_label = label.split()[0]
        # plt.savefig(os.path.join(OUTPUT_DIR, f"rh2_fairness_{safe_label}.png"))
        plt.figure(figsize=(8, 6))
        df_melt = sub_df.melt(value_vars=metrics, var_name='Workflow', value_name='Score')
        df_melt['Workflow'] = df_melt['Workflow'].map({
            'W1_Fair_Overall':'W1', 'W2_Fair_Overall':'W2', 'W3_Fair_Overall':'W3'
        })
        df_melt['Workflow'] = enforce_workflow_order(df_melt['Workflow'])

        sns.boxplot(
            x="Workflow", y="Score", data=df_melt,
            order=WORKFLOW_ORDER,
            palette=WORKFLOW_COLORS,
            saturation=1,              # <-- key: prevents seaborn from muting colors
            showmeans=True, meanline=True,
            meanprops={"color": "black", "linewidth": 1.5, "linestyle": "--"}
        )

        plt.title(f"RH2: Fairness ({label})")
        plt.ylim(1, 5.5)
        plt.tight_layout()
        safe_label = label.split()[0]
        plt.savefig(os.path.join(OUTPUT_DIR, f"rh2_fairness_{safe_label}.png"))

        print(f"Saved plot: rh2_fairness_{safe_label}.png")

# ---------------------------------------------------------
# RH3: Cost Preference (Using Efficiency Choice as Proxy)
# NOTE: User updated mapping. Choice_Effic_1/2 are about Efficiency.
# We will assume "Efficiency" implies "Low Cost" preference.
# ---------------------------------------------------------
def analyze_rh3_efficiency(df):
    print("\n" + "="*60)
    print("RH3: Efficiency/Cost Preference (Split by AI Expertise)")
    print("="*60)
    
    # Using BOTH choice columns now
    choice_cols = ['Choice_Effic_1', 'Choice_Effic_2']
    
    df_novice = df[df['Exp_AI_Score'] == 1]
    df_expert = df[df['Exp_AI_Score'] >= 2]
    
    cost_map = {'Workflow 1': 'W1', 'Workflow 2': 'W2', 'Workflow 3': 'W3'}
    
    groups = [("Non-Experts (Score=1)", df_novice), ("Experts (Score>=2)", df_expert)]
    
    for label, sub_df in groups:
        print(f"\n--- Group: {label} (N={len(sub_df)}) ---")
        if len(sub_df) < 1: continue
        
        # Combine choices from both columns into a single series
        # Stack allows us to treat each choice as a separate observation
        combined_choices = sub_df[choice_cols].stack().reset_index(drop=True)
        mapped_choices = combined_choices.map(cost_map)
        
        counts = mapped_choices.value_counts()
        

        print(counts)
        
        cats = ['W1', 'W2', 'W3']
        counts = counts.reindex(WORKFLOW_ORDER, fill_value=0)

        ax = counts.plot(
            kind="bar",
            color=[WORKFLOW_COLORS[w] for w in WORKFLOW_ORDER]
        )
        
        # Chi-Square
        # We now have 2 observations per user, so N is effectively doubled for the test of preferences
        if len(combined_choices) >= 5:
            try:
                obs = counts.values
                stat, p = stats.chisquare(obs)
                print(f"Chi-Square (vs Uniform): p={p:.4f}")
            except: pass
            
        plt.figure(figsize=(8, 6))
        counts = counts.reindex(WORKFLOW_ORDER, fill_value=0)

        ax = counts.plot(
            kind="bar",
            color=[WORKFLOW_COLORS[w] for w in WORKFLOW_ORDER]
        )
        plt.title(f"RH3: Efficiency Preference ({label})")
        plt.ylabel("Total Count (Both Examples)")
        plt.xticks(rotation=0)
        plt.tight_layout()
        safe_label = label.split()[0]
        plt.savefig(os.path.join(OUTPUT_DIR, f"rh3_effic_{safe_label}.png"))
        # plt.figure(figsize=(8, 6))
        # counts.plot(kind='bar', color=['green', 'red', 'orange'])
        # plt.title(f"RH3: Efficiency Preference ({label})")
        # plt.ylabel("Total Count (Both Examples)")
        # plt.xticks(rotation=0)
        # plt.tight_layout()
        # safe_label = label.split()[0]
        # plt.savefig(os.path.join(OUTPUT_DIR, f"rh3_effic_{safe_label}.png"))
        print(f"Saved plot: rh3_effic_{safe_label}.png")

# ---------------------------------------------------------
# Automated Agreement (Analysis of all 3 types)
# ---------------------------------------------------------
def analyze_automated_agreement(df):
    print("\n" + "="*60)
    print("Automated Method Agreement Analysis")
    print("="*60)
    
    # Updated Agreement Columns
    agree_cols = {
        "Auto_Agree_Corr": "Correctness Panel",
        "Auto_Agree_Fair": "Fairness Panel",
        "Auto_Agree_Effic": "Efficiency Panel" # New
    }
    
    # 1. Stats Table
    stats_rows = []
    for col, name in agree_cols.items():
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            t_stat, p_val = stats.ttest_1samp(df[col].dropna(), 3.0)
            stats_rows.append({
                "Metric": name, "Mean": mean, "Std": std, "p_val (vs 3.0)": p_val
            })
    
    res_df = pd.DataFrame(stats_rows)
    print(res_df)
    
    # 2. Combined Plot (Boxplot of the 3 metrics)
    plt.figure(figsize=(10, 6))
    df_melt = df.melt(value_vars=list(agree_cols.keys()), var_name='Metric_Code', value_name='Score')
    df_melt['Metric'] = df_melt['Metric_Code'].map(agree_cols)
    
    sns.boxplot(
        x='Metric', y='Score', data=df_melt, palette="Pastel1",
        showmeans=True, meanline=True,
        meanprops={"color": "red", "linewidth": 1.5, "linestyle": "--"}
    )
    plt.title("Agreement with Automated Ranking (Across Phases)")
    plt.ylabel("Likert Score (1-5)")
    plt.ylim(1, 5.5)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "auto_agreement_detailed.png"))
    print("Saved plot: auto_agreement_detailed.png")

def main():
    df = load_and_prep_data()
    
    analyze_rh1_correctness(df)
    analyze_rh2_fairness(df)
    analyze_rh3_efficiency(df) # Renamed to efficiency
    analyze_automated_agreement(df)
    
    print(f"\nDone. Results in '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()