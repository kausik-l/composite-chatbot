import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import time

# Add root to path so imports work
root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(root)

from env.chatbot_env import ChatbotPipelineEnv
from planner.policy import ChatbotContextAwareQPlanner
from planner.baselines import ChatbotRandomPipelinePlanner, ChatbotFixedPipelinePlanner

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "ACTIVE_MODES": ["WRS", "DIE", "BOTH"], 
    "TRAIN_EPISODES": 100, 
    "EVAL_EPISODES": 500,
    "DATA_DIR": os.path.join(root,"..", "data", "outcome"),
    "DOMAIN_PATH": os.path.join(root, "domain", "chatbot.rddl"),
    "INSTANCE_PATH": os.path.join(root, "instances", "chatbot_instance.rddl"),
    "LOG_FILE": "chatbot_evaluation_log.csv"
}

# Mapping Stages to Components
# UPDATED to include sum1, sum2, no_sum
STAGE_MAP = {
    "s1": ["para_none", "para_spanish", "para_danish"],
    "s2": ["sys_s1", "sys_s2", "sys_s3"],
    "s3": ["sum_sum1", "sum_sum2", "sum_no"]
}

def run_experiment_full_trace():
    print(f"=== Chatbot Pipeline Experiment ===")
    
    if not os.path.exists(CONFIG["DATA_DIR"]):
        print(f"Error: Data directory {CONFIG['DATA_DIR']} not found.")
        os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
        return

    # 1. Build Agents
    agents = []
    
    # Q-Learning Agents
    colors = {"WRS": "blue", "DIE": "orange", "BOTH": "green"}
    for mode in CONFIG["ACTIVE_MODES"]:
        agents.append({
            "name": f"Q-Learning ({mode})",
            "agent": ChatbotContextAwareQPlanner(action_space=None, stage_map=STAGE_MAP, alpha=0.1, gamma=0.99, epsilon=0.2),
            "learns": True,
            "train_mode": mode,
            "color": colors.get(mode, "black")
        })

    # Baselines
    agents.extend([
        {"name": "Fixed (Default)", "agent": ChatbotFixedPipelinePlanner(STAGE_MAP, selection_index=0), "learns": False, "train_mode": "BOTH", "color": "purple"},
        {"name": "Random", "agent": ChatbotRandomPipelinePlanner(STAGE_MAP), "learns": False, "train_mode": "BOTH", "color": "gray"}
    ])
    
    if os.path.exists(CONFIG["LOG_FILE"]): os.remove(CONFIG["LOG_FILE"])

    results_table = []
    
    # Store training history for plotting
    plot_data = {} 

    # 2. Execution Loop
    for entry in agents:
        name = entry['name']
        agent = entry['agent']
        learns = entry['learns']
        train_mode = entry['train_mode']
        color = entry.get('color', 'black')
        
        print(f"\n--- Processing Agent: {name} ---")
        
        # A. TRAINING PHASE
        if learns:
            print(f"  > Training with Reward Mode: {train_mode}")
            
            # Use a specific env for training to capture metrics cleanly
            train_env = ChatbotPipelineEnv(
                CONFIG["DOMAIN_PATH"], CONFIG["INSTANCE_PATH"], CONFIG["DATA_DIR"], reward_mode=train_mode
            )
            
            # History lists for this agent
            h_reward, h_wrs, h_die = [], [], []
            
            for _ in tqdm(range(CONFIG["TRAIN_EPISODES"]), desc="Train"):
                state, _ = train_env.reset()
                ep_reward = 0
                ep_final_wrs = 0.0
                ep_final_die = 0.0
                
                while True:
                    action = agent.sample_action(state)
                    next_state, r, done, _, info = train_env.step(action)
                    agent.update(state, action, r, next_state)
                    state = next_state
                    ep_reward += r
                    
                    # Capture metrics if present (usually at end of pipeline)
                    if 'metrics' in info:
                        ep_final_wrs = max(ep_final_wrs, info['metrics'].get('raw_wrs', 0.0))
                        ep_final_die = max(ep_final_die, info['metrics'].get('raw_die', 0.0))
                    
                    if done: break
                
                h_reward.append(ep_reward)
                h_wrs.append(ep_final_wrs)
                h_die.append(ep_final_die)
            
            # Store for plotting (FIXED: Using Dictionary)
            plot_data[name] = {
                "Reward": h_reward,
                "WRS": h_wrs,
                "DIE": h_die,
                "Color": color
            }
            
            # Freeze agent for eval
            agent.epsilon = 0.0

        # B. EVALUATION PHASE
        print(f"  > Evaluating under 'BOTH' mode...")
        # Re-init env to ensure clean state and uniform evaluation criteria
        eval_env = ChatbotPipelineEnv(
            CONFIG["DOMAIN_PATH"], CONFIG["INSTANCE_PATH"], CONFIG["DATA_DIR"], reward_mode="BOTH"
        )
        
        current_agent_trace = []
        eval_rewards, eval_costs, eval_wrs, eval_die, eval_comp, eval_qual = [], [], [], [], [], []
        pipeline_choices = []
        
        for ep_idx in tqdm(range(CONFIG["EVAL_EPISODES"]), desc=f"Eval {name}"):
            state, _ = eval_env.reset()
            stage_count = 1
            ep_reward, ep_cost, ep_final_wrs, ep_final_die, ep_final_comp, ep_final_qual = 0, 0, 0, 0, 0, 0
            current_pipeline = []
            
            while True:
                action = agent.sample_action(state)
                
                selected_model_name = "None"
                for k, v in action.items():
                    if v == 1 and "select_component" in k:
                        selected_model_name = k.split("___")[-1]
                        current_pipeline.append(selected_model_name)
                        break

                next_state, r, done, _, info = eval_env.step(action)
                state = next_state
                ep_reward += r
                
                metrics = info.get('metrics', {})
                ep_cost += metrics.get('rddl_cost', 0.0)
                ep_final_wrs = max(ep_final_wrs, metrics.get('raw_wrs', 0.0))
                ep_final_die = max(ep_final_die, metrics.get('raw_die', 0.0))
                ep_final_comp = max(ep_final_comp, metrics.get('compression_reward', 0.0))
                ep_final_qual = max(ep_final_qual, metrics.get('quality_reward', 0.0))
                
                current_agent_trace.append({
                    "Agent": name,
                    "Episode": ep_idx,
                    "Stage": stage_count,
                    "Action": selected_model_name,
                    "Step_Reward": r,
                    "Step_Cost": metrics.get('rddl_cost', 0.0),
                    "Final_WRS": metrics.get('raw_wrs', 0.0),
                    "Final_DIE": metrics.get('raw_die', 0.0),
                    "Final_Comp_Reward": metrics.get('compression_reward', 0.0),
                    "Final_Quality": metrics.get('quality_reward', 0.0)
                })
                stage_count += 1
                if done: break
            
            eval_rewards.append(ep_reward)
            eval_costs.append(ep_cost)
            eval_wrs.append(ep_final_wrs)
            eval_die.append(ep_final_die)
            eval_comp.append(ep_final_comp)
            eval_qual.append(ep_final_qual)
            pipeline_choices.append(tuple(current_pipeline))

        # Save Trace
        write_header = not os.path.exists(CONFIG["LOG_FILE"]) or os.path.getsize(CONFIG["LOG_FILE"]) == 0
        df_chunk = pd.DataFrame(current_agent_trace)
        if not df_chunk.empty:
            df_chunk.to_csv(CONFIG["LOG_FILE"], mode='a', header=write_header, index=False)

        # Top 3 Pipelines Logic
        top3_str = "N/A"
        if pipeline_choices:
            counts = Counter(pipeline_choices).most_common(3)
            parts = []
            for pipe, count in counts:
                pipe_str = " -> ".join(pipe)
                parts.append(f"{pipe_str} ({count})")
            top3_str = "; ".join(parts)

        # Result Row
        results_table.append({
            "Agent": name,
            "Train Mode": train_mode if learns else "N/A",
            "Avg Reward": np.mean(eval_rewards),
            "Avg Cost": np.mean(eval_costs),
            "Avg WRS": np.mean(eval_wrs),
            "Avg DIE": np.mean(eval_die),
            "Avg Quality": np.mean(eval_qual),
            "Avg Compression": np.mean(eval_comp),
            "Top Pipeline": top3_str
        })

    # 3. Output Table
    df = pd.DataFrame(results_table)
    print("\n=== FINAL CHATBOT RESULTS ===")
    print(df[["Agent", "Avg Reward", "Avg Quality", "Avg Compression", "Avg WRS", "Avg DIE", "Top Pipeline"]].to_string(index=False))
    df.to_csv("chatbot_results_summary.csv", index=False)
    
    # 4. Generate Multi-Panel Training Plot
    if plot_data:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Helper to smooth lines
        def smooth_data(data, window=20):
            if len(data) > window:
                return np.convolve(data, np.ones(window)/window, mode='valid')
            return data

        # Plot 1: Total Reward
        for name, data in plot_data.items():
            rew = data["Reward"]
            smooth = smooth_data(rew)
            axes[0].plot(smooth, label=name, color=data["Color"])
        axes[0].set_title("Total Reward (Smoothed)")
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Reward")
        axes[0].legend()

        # Plot 2: WRS Metric
        for name, data in plot_data.items():
            wrs = data["WRS"]
            smooth = smooth_data(wrs)
            axes[1].plot(smooth, label=name, color=data["Color"])
        axes[1].set_title("WRS Metric (Smoothed)")
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("WRS Value")

        # Plot 3: DIE Metric
        for name, data in plot_data.items():
            die = data["DIE"]
            smooth = smooth_data(die)
            axes[2].plot(smooth, label=name, color=data["Color"])
        axes[2].set_title("DIE Metric (Smoothed)")
        axes[2].set_xlabel("Episode")
        axes[2].set_ylabel("DIE Value")

        plt.tight_layout()
        plt.savefig("chatbot_training_metrics.png")
        print("\nSaved multi-panel training plots to 'chatbot_training_metrics.png'")

if __name__ == "__main__":
    run_experiment_full_trace()