import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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
    "TRAIN_EPISODES": 200, 
    "EVAL_EPISODES": 50,
    "DATA_DIR": os.path.join(root, "data", "outcome"),
    "DOMAIN_PATH": os.path.join(root, "domain", "chatbot.rddl"),
    "INSTANCE_PATH": os.path.join(root, "instances", "chatbot_instance.rddl"),
    "LOG_FILE": "chatbot_evaluation_log.csv"
}

# Mapping Stages to Components (must match RDDL)
STAGE_MAP = {
    "s1": ["para_none", "para_spanish", "para_danish"],
    "s2": ["sys_s1", "sys_s2", "sys_s3"],
    "s3": ["sum_yes", "sum_no"]
}

def run_experiment():
    print(f"=== Chatbot Pipeline Experiment ===")
    
    if not os.path.exists(CONFIG["DATA_DIR"]):
        print(f"Error: Data directory {CONFIG['DATA_DIR']} not found.")
        os.makedirs(CONFIG["DATA_DIR"], exist_ok=True)
        print("Created empty data directory. Please put CSV files there.")

    results_table = []
    
    # Global log list to store all episode details
    global_logs = []

    for mode in CONFIG["ACTIVE_MODES"]:
        print(f"\n--- Evaluation for Reward Mode: {mode} ---")
        
        # 1. Setup Environment
        env = ChatbotPipelineEnv(
            domain_file=CONFIG["DOMAIN_PATH"], 
            instance_file=CONFIG["INSTANCE_PATH"], 
            data_dir=CONFIG["DATA_DIR"],
            reward_mode=mode
        )
        
        # 2. Define Agents
        agents = {}
        
        # Q-Learning Agent
        q_agent = ChatbotContextAwareQPlanner(
            action_space=None, 
            stage_map=STAGE_MAP,
            alpha=0.1, gamma=0.99, epsilon=0.2
        )
        agents["Q-Learning"] = q_agent

        # Random Baseline
        agents["Random"] = ChatbotRandomPipelinePlanner(STAGE_MAP)

        # Fixed Baseline (Default: None -> S1 -> No Sum)
        agents["Fixed (Default)"] = ChatbotFixedPipelinePlanner(STAGE_MAP, selection_index=0)
        
        # 3. Train Q-Learning Agent
        print(f"Training Q-Learning Agent ({mode})...")
        train_rewards = []
        for _ in tqdm(range(CONFIG["TRAIN_EPISODES"]), desc="Training"):
            obs, _ = env.reset()
            done = False
            ep_reward = 0
            while not done:
                action = q_agent.sample_action(obs)
                next_obs, reward, done, _, _ = env.step(action)
                q_agent.update(obs, action, reward, next_obs)
                obs = next_obs
                ep_reward += reward
            train_rewards.append(ep_reward)
        
        # Switch Q-Agent to Greedy for Eval
        q_agent.epsilon = 0.0

        # 4. Evaluate ALL Agents
        for agent_name, agent in agents.items():
            eval_rewards = []
            raw_wrs_list = []
            raw_die_list = []
            pipeline_counts = {}

            # Run Eval Episodes
            for i in range(CONFIG["EVAL_EPISODES"]):
                obs, _ = env.reset()
                done = False
                ep_reward = 0
                path = []
                current_metrics = {}

                while not done:
                    action = agent.sample_action(obs)
                    
                    # Record path
                    for k, v in action.items():
                        if v == 1: path.append(k.split("___")[-1])

                    next_obs, reward, done, _, info = env.step(action)
                    # No update during eval for Q-Learning
                    obs = next_obs
                    ep_reward += reward
                    
                    if done and 'metrics' in info:
                        current_metrics = info['metrics']

                eval_rewards.append(ep_reward)
                
                # Get metrics with defaults
                wrs = current_metrics.get('raw_wrs', 0.0)
                die = current_metrics.get('raw_die', 0.0)
                cost = current_metrics.get('rddl_cost', 0.0)
                
                raw_wrs_list.append(wrs)
                raw_die_list.append(die)
                
                path_str = " -> ".join(path)
                pipeline_counts[path_str] = pipeline_counts.get(path_str, 0) + 1
                
                # LOGGING
                global_logs.append({
                    "Reward Mode": mode,
                    "Agent": agent_name,
                    "Episode": i,
                    "Total Reward": ep_reward,
                    "WRS": wrs,
                    "DIE": die,
                    "Cost": cost,
                    "Pipeline": path_str
                })

            # Stats
            top_pipelines = sorted(pipeline_counts.items(), key=lambda x: x[1], reverse=True)[:1]
            top_str = top_pipelines[0][0] if top_pipelines else "N/A"
            
            row = {
                "Reward Mode": mode,
                "Agent": agent_name,
                "Avg Reward": np.mean(eval_rewards),
                "Avg WRS": np.mean(raw_wrs_list),
                "Avg DIE": np.mean(raw_die_list),
                "Top Pipeline": top_str
            }
            results_table.append(row)
        
        # Plot Training Curve for Q-Learning
        plt.plot(train_rewards, label=f"{mode}")

    # Final Output
    plt.title("Q-Learning Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.legend()
    plt.savefig("chatbot_training_curve.png")
    
    # Save Results Table
    df_res = pd.DataFrame(results_table)
    df_res = df_res.sort_values(by=["Reward Mode", "Avg Reward"], ascending=[True, False])
    print("\n=== FINAL CHATBOT RESULTS ===")
    print(df_res.to_string(index=False))
    df_res.to_csv("chatbot_results.csv", index=False)
    
    # Save Detailed Log
    log_filename = CONFIG.get("LOG_FILE", "chatbot_evaluation_log.csv")
    df_log = pd.DataFrame(global_logs)
    df_log.to_csv(log_filename, index=False)
    print(f"\nDetailed evaluation log saved to {log_filename}")

if __name__ == "__main__":
    run_experiment()