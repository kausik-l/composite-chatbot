import pandas as pd
import numpy as np

# Configuration
LOG_FILE = "code/chatbot_evaluation_log.csv"

# The specific workflows to analyze
TARGET_WORKFLOWS = [
    "para_none -> sys_s2 -> sum_no",
    "para_spanish -> sys_s2 -> sum_no",
    "para_none -> sys_s1 -> sum_sum2"
]

def analyze_workflows():
    print(f"Loading log file: {LOG_FILE}...")
    try:
        df = pd.read_csv(LOG_FILE)
    except FileNotFoundError:
        print("Error: Log file not found.")
        return

    # 1. Group by Episode (and Agent) to reconstruct workflows
    # We assume 'Episode' is unique per Agent run, so grouping by Agent+Episode is safest.
    # However, since episode IDs restart for each agent, we must group by Agent first.
    
    results = []

    # Iterate through unique agent runs to avoid mixing up episode 0 from Agent A and Agent B
    for agent_name in df['Agent'].unique():
        agent_df = df[df['Agent'] == agent_name]
        
        # Group by Episode to aggregate steps
        grouped = agent_df.groupby('Episode')
        
        for ep_id, group in grouped:
            # Sort by Stage to ensure correct order (1 -> 2 -> 3)
            group = group.sort_values('Stage')
            
            # Reconstruct Workflow String
            actions = group['Action'].tolist()
            # Filter out 'None' or empty actions if any
            actions = [a for a in actions if a and a != "None"]
            workflow_str = " -> ".join(actions)
            
            if workflow_str in TARGET_WORKFLOWS:
                # Extract Metrics
                # Metrics like WRS/DIE/Quality are logged at the final stage (Stage 3 usually)
                # or accumulated. Based on log format, they appear in the final row.
                final_row = group.iloc[-1]
                
                # Cost is accumulated across stages
                total_cost = group['Step_Cost'].sum()
                
                # Quality and Fairness are snapshot values at the end
                quality = final_row.get('Final_Quality', 0.0)
                wrs = final_row.get('Final_WRS', 0.0)
                die = final_row.get('Final_DIE', 0.0)
                
                # Combined Penalty Score
                combined_penalty = wrs + die
                
                results.append({
                    "Workflow": workflow_str,
                    "Combined_Penalty": combined_penalty,
                    "Quality": quality,
                    "Cost": total_cost
                })

    # 2. Compute Statistics
    if not results:
        print("No matching workflows found in the log.")
        return

    result_df = pd.DataFrame(results)
    
    print("\n" + "="*60)
    print(f"{'WORKFLOW':<40} | {'METRIC':<15} | {'MEAN':<10} | {'STD':<10}")
    print("="*60)

    for workflow in TARGET_WORKFLOWS:
        # Filter for this specific workflow
        subset = result_df[result_df['Workflow'] == workflow]
        
        if subset.empty:
            print(f"{workflow:<40} | {'(No Data)':<15} | {'-':<10} | {'-':<10}")
            continue
            
        # Calculate Stats
        metrics = ["Combined_Penalty", "Quality", "Cost"]
        
        for metric in metrics:
            mean_val = subset[metric].mean()
            std_val = subset[metric].std()
            
            # Print row
            # Only print workflow name for the first metric to keep it clean
            wf_label = workflow if metric == metrics[0] else ""
            print(f"{wf_label:<40} | {metric:<15} | {mean_val:<10.4f} | {std_val:<10.4f}")
        print("-" * 60)

if __name__ == "__main__":
    analyze_workflows()