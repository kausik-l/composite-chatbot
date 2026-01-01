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
    "TRAIN_EPISODES": 20, 
    "EVAL_EPISODES": 500,
    "DATA_DIR": os.path.join(root,"..", "data", "outcome"),
    "DOMAIN_PATH": os.path.join(root, "domain", "chatbot.rddl"),
    "INSTANCE_PATH": os.path.join(root, "instances", "chatbot_instance.rddl"),
    "LOG_FILE": "chatbot_evaluation_log.csv"
}

# Mapping Stages to Components
STAGE_MAP = {
    "s1": ["para_none", "para_spanish", "para_danish"],
    "s2": ["sys_s1", "sys_s2", "sys_s3"],
    "s3": ["sum_yes", "sum_no"]
}

def run_experiment_full_trace():
    print(f"=== Chatbot Pipeline Experiment (Structure: Sentiment Large) ===")
    
    if not os.path.exists(CONFIG["DATA_DIR"]):
        print(f"Error: Data directory {CONFIG['DATA_DIR']} not found.")
        os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
        print("Created empty data directory. Please put CSV files there.")
        return

    # Initialize Env just for reading dimensions/setup if needed
    env = ChatbotPipelineEnv(
        domain_file=CONFIG["DOMAIN_PATH"], 
        instance_file=CONFIG["INSTANCE_PATH"], 
        data_dir=CONFIG["DATA_DIR"], 
        reward_mode="BOTH"
    )

    # 1. Build Agents List
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

    # Baselines (Evaluated under BOTH mode)
    agents.extend([
        {"name": "Fixed (Default)", "agent": ChatbotFixedPipelinePlanner(STAGE_MAP, selection_index=0), "learns": False, "train_mode": "BOTH", "color": "purple"},
        {"name": "Random", "agent": ChatbotRandomPipelinePlanner(STAGE_MAP), "learns": False, "train_mode": "BOTH", "color": "gray"}
    ])
    
    # Setup Trace File
    if os.path.exists(CONFIG["LOG_FILE"]):
        os.remove(CONFIG["LOG_FILE"])

    results_table = []
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
            env.reward_mode = train_mode
            
            rewards = []
            for _ in tqdm(range(CONFIG["TRAIN_EPISODES"]), desc="Train"):
                state, _ = env.reset()
                ep_reward = 0
                while True:
                    action = agent.sample_action(state)
                    next_state, r, done, _, _ = env.step(action)
                    agent.update(state, action, r, next_state)
                    state = next_state
                    ep_reward += r
                    if done: break
                rewards.append(ep_reward)
            
            # Store training curve
            plot_data[name] = (rewards, color)
            
            # Freeze agent for eval
            agent.epsilon = 0.0

        # B. EVALUATION PHASE (Always 'BOTH' for apples-to-apples)
        print(f"  > Evaluating under 'BOTH' mode...")
        env.reward_mode = "BOTH"
        
        current_agent_trace = []
        eval_rewards = []
        eval_costs = []
        eval_wrs = []
        eval_die = []
        pipeline_choices = []
        
        for ep_idx in tqdm(range(CONFIG["EVAL_EPISODES"]), desc=f"Eval {name}"):
            state, _ = env.reset()
            stage_count = 1
            
            # Accumulators for this episode
            ep_reward = 0
            ep_cost = 0
            ep_final_wrs = 0
            ep_final_die = 0
            
            current_pipeline = []
            
            while True:
                action = agent.sample_action(state)
                
                # Track pipeline path
                selected_model_name = "None"
                for k, v in action.items():
                    if v == 1 and "select_component" in k:
                        selected_model_name = k.split("___")[-1]
                        current_pipeline.append(selected_model_name)
                        break

                next_state, r, done, _, info = env.step(action)
                state = next_state
                
                ep_reward += r
                
                # ACCUMULATE METRICS STEP-BY-STEP (Like Sentiment script)
                metrics = info.get('metrics', {})
                ep_cost += metrics.get('rddl_cost', 0.0)
                
                # WRS and DIE only appear at the final step, but safe to overwrite/add 
                # since they are 0.0 in non-final steps in this env.
                ep_final_wrs = max(ep_final_wrs, metrics.get('raw_wrs', 0.0))
                ep_final_die = max(ep_final_die, metrics.get('raw_die', 0.0))
                
                # Log Trace
                current_agent_trace.append({
                    "Agent": name,
                    "Episode": ep_idx,
                    "Stage": stage_count,
                    "Action": selected_model_name,
                    "Step_Reward": r,
                    "Step_Cost": metrics.get('rddl_cost', 0.0),
                    "Final_WRS": metrics.get('raw_wrs', 0.0),
                    "Final_DIE": metrics.get('raw_die', 0.0)
                })
                
                stage_count += 1
                if done: break
            
            eval_rewards.append(ep_reward)
            eval_costs.append(ep_cost)
            eval_wrs.append(ep_final_wrs)
            eval_die.append(ep_final_die)
            pipeline_choices.append(tuple(current_pipeline))

        # Save Trace incrementally
        write_header = not os.path.exists(CONFIG["LOG_FILE"]) or os.path.getsize(CONFIG["LOG_FILE"]) == 0
        df_chunk = pd.DataFrame(current_agent_trace)
        if not df_chunk.empty:
            df_chunk.to_csv(CONFIG["LOG_FILE"], mode='a', header=write_header, index=False)

        # Top Pipeline
        top3_str = "N/A"
        if pipeline_choices:
            counts = Counter(pipeline_choices).most_common(1)
            pipe, count = counts[0]
            pipe_str = " -> ".join(pipe)
            top3_str = f"{pipe_str} ({count})"

        # Store Result
        row = {
            "Agent": name,
            "Train Mode": train_mode if learns else "N/A",
            "Avg Reward": np.mean(eval_rewards),
            "Avg Cost": np.mean(eval_costs),
            "Avg WRS": np.mean(eval_wrs),
            "Avg DIE": np.mean(eval_die),
            "Top Pipeline": top3_str
        }
        results_table.append(row)

    # 3. Output Table
    df = pd.DataFrame(results_table)
    print("\n=== FINAL CHATBOT RESULTS ===")
    cols = ["Agent", "Avg Reward", "Avg Cost", "Avg WRS", "Avg DIE", "Top Pipeline"]
    print(df[cols].to_string(index=False))
    df.to_csv("chatbot_results_summary.csv", index=False)
    
    # 4. Plot Training
    if plot_data:
        plt.figure(figsize=(10, 6))
        for name, (rew, col) in plot_data.items():
            smoothed = np.convolve(rew, np.ones(20)/20, mode='valid') if len(rew)>20 else rew
            plt.plot(smoothed, label=name, color=col)
        plt.title("Q-Learning Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.legend()
        plt.savefig("chatbot_training_curve.png")
        print("\nSaved training curve to 'chatbot_training_curve.png'")

if __name__ == "__main__":
    run_experiment_full_trace()