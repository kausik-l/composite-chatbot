import pandas as pd
import numpy as np
from scipy import stats
import os

# Configuration
# Path to your evaluation log
LOG_FILE = "code/chatbot_evaluation_log.csv"
SCENARIO_NAME = "Chatbot Optimization (3 Stages)"
NUM_STAGES = 3 # para -> sys -> sum

def analyze_significance(file_path, scenario_name):
    print(f"\n{'='*80}")
    print(f"ANALYZING: {scenario_name}")
    print(f"Source: {file_path}")
    print(f"{'='*80}")

    if not os.path.exists(file_path):
        print(f"[ERROR] File not found: {file_path}")
        return

    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"[ERROR] Could not read CSV: {e}")
        return

    # 1. DATA CLEANING
    # Ensure we only look at valid steps
    df = df.dropna(subset=['Step_Reward'])

    # 2. AGGREGATION (Episode Totals)
    # We sum the rewards for each episode to get Independent Samples (N=50)
    # Group by Agent and Episode
    episode_stats = df.groupby(['Agent', 'Episode'])['Step_Reward'].sum().reset_index()

    # 3. NORMALIZE (Reward Per Stage) for easier reading, or keep total
    episode_stats['Reward_Per_Stage'] = episode_stats['Step_Reward'] / NUM_STAGES

    # 4. IDENTIFY CHAMPION (Auto-Detect by Mean Reward)
    # We want to find the Q-Learning agent with the highest mean reward
    # Filter for Q-Learning agents
    q_agents = episode_stats[episode_stats['Agent'].str.contains("Q-Learning")]
    
    if q_agents.empty:
        # Fallback if no Q-Learning agents found (e.g. only baselines)
        champion = episode_stats.groupby('Agent')['Reward_Per_Stage'].mean().idxmax()
    else:
        # Pick the best Q-Learning agent
        champion = q_agents.groupby('Agent')['Reward_Per_Stage'].mean().idxmax()

    # Get Champion Data
    champ_scores = episode_stats[episode_stats['Agent'] == champion]['Reward_Per_Stage']
    champ_mean = champ_scores.mean()
    champ_std = champ_scores.std()

    print(f"\n>>> CHAMPION AGENT: {champion}")
    print(f"    Mean Reward (Per Stage): {champ_mean:.4f}")
    print(f"    Std Dev:                 {champ_std:.4f}")
    print(f"    Sample Size (N):         {len(champ_scores)}")

    print(f"\n{'-'*30} STATISTICAL COMPARISONS {'-'*30}")
    print(f"{'OPPONENT':<30} | {'MEAN':<10} | {'DIFF':<10} | {'P-VALUE':<10} | {'SIG'}")
    print("-" * 80)

    # Sort opponents: Other Q-Learners first, then Baselines
    available_agents = episode_stats['Agent'].unique()
    others = [a for a in available_agents if a != champion]
    
    # Sort logic: Q-Learning top, others bottom
    others.sort(key=lambda x: "Q-Learning" not in x) 

    for opponent in others:
        opp_scores = episode_stats[episode_stats['Agent'] == opponent]['Reward_Per_Stage']
        opp_mean = opp_scores.mean()
        
        # --- WELCH'S T-TEST ---
        # equal_var=False handles the fact that Heuristic/Random often have higher variance
        t_stat, p_val = stats.ttest_ind(champ_scores, opp_scores, equal_var=False)
        
        # Calculate Difference
        diff = champ_mean - opp_mean
        
        # Improvement Percentage
        # Handle zero division or weird signs
        if opp_mean == 0:
            pct_imp = float('inf') if diff > 0 else 0.0
        else:
            pct_imp = (diff / abs(opp_mean)) * 100

        # Significance Markers
        sig = "ns"
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"

        # Formatting
        result_str = f"{opponent:<30} | {opp_mean:<10.4f} | {diff:<+10.4f} | {p_val:.2e}   | {sig}"
        print(result_str)
        
        if p_val < 0.05:
            if diff > 0:
                print(f"    -> RESULT: Champion is BETTER by {pct_imp:.1f}% (Significant)")
            else:
                print(f"    -> RESULT: Champion is WORSE by {abs(pct_imp):.1f}% (Significant)")

    print(f"{'='*80}\n")

if __name__ == "__main__":
    analyze_significance(LOG_FILE, SCENARIO_NAME)