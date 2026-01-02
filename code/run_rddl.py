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
from planner.baselines import ChatbotRandomPipelinePlanner, ChatbotFixedPipelinePlanner, ChatbotLookaheadFairnessPlanner

# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    "ACTIVE_MODES": ["WRS", "DIE", "BOTH"], 
    "TRAIN_EPISODES": 200, 
    "EVAL_EPISODES": 500,
    "EVAL_TEMPERATURE": 0.1, 
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
    # Note: Lookahead needs env instance, we will pass eval_env later or init a dummy one?
    # Actually, we can't pass eval_env yet because it's created in the loop.
    # We will instantiate Lookahead inside the loop or pass a shared env.
    
    # Let's add them as (Name, Class, Params) tuples to init later
    baselines_meta = [
        ("Fixed (Default)", ChatbotFixedPipelinePlanner, {"selection_index": 0}),
        ("Random", ChatbotRandomPipelinePlanner, {}),
        ("Heuristic (WRS Lookahead)", ChatbotLookaheadFairnessPlanner, {"env_needed": True})
    ]
    
    if os.path.exists(CONFIG["LOG_FILE"]): os.remove(CONFIG["LOG_FILE"])

    results_table = []
    plot_data = {} 
    eval_plot_data = {} # Initialize eval_plot_data

    # 2. Execution Loop
    # First process Q-Agents (Training)
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
            train_env = ChatbotPipelineEnv(
                CONFIG["DOMAIN_PATH"], CONFIG["INSTANCE_PATH"], CONFIG["DATA_DIR"], reward_mode=train_mode
            )
            
            rewards = []
            h_wrs, h_die = [], []

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
                    
                    if 'metrics' in info:
                        ep_final_wrs = max(ep_final_wrs, info['metrics'].get('raw_wrs', 0.0))
                        ep_final_die = max(ep_final_die, info['metrics'].get('raw_die', 0.0))

                    if done: break
                
                rewards.append(ep_reward)
                h_wrs.append(ep_final_wrs)
                h_die.append(ep_final_die)
            
            # Store as dictionary for plotting access
            plot_data[name] = {
                "Reward": rewards,
                "WRS": h_wrs,
                "DIE": h_die,
                "Color": color
            }
            agent.epsilon = 0.0

    # 3. EVALUATION PHASE (For ALL Agents including Baselines)
    print(f"\n--- Starting Unified Evaluation (Mode: BOTH) ---")
    
    # Prepare Evaluation Environment
    eval_env = ChatbotPipelineEnv(
        CONFIG["DOMAIN_PATH"], CONFIG["INSTANCE_PATH"], CONFIG["DATA_DIR"], reward_mode="BOTH"
    )
    
    # Add baselines to the list for evaluation
    eval_agents_list = []
    
    # Add trained Q-Agents
    for entry in agents:
        eval_agents_list.append((entry['name'], entry['agent'], entry['color']))
        
    # Add/Init Baselines with colors
    baseline_colors = {"Fixed (Default)": "purple", "Random": "gray", "Heuristic (WRS Lookahead)": "red"}
    
    for name, cls, params in baselines_meta:
        if params.get("env_needed"):
            # Pass eval_env to Lookahead
            agent = cls(STAGE_MAP, env=eval_env)
        else:
            agent = cls(STAGE_MAP, **params)
        eval_agents_list.append((name, agent, baseline_colors.get(name, "black")))

    # Run Eval Loop
    for agent_name, agent, color in eval_agents_list:
        print(f"Evaluating {agent_name}...")
        
        current_agent_trace = []
        # Metrics history for plotting
        hist_reward, hist_wrs, hist_die, hist_comp = [], [], [], []
        
        eval_rewards, eval_costs, eval_wrs, eval_die, eval_comp, eval_qual = [], [], [], [], [], []
        pipeline_choices = []
        
        for ep_idx in tqdm(range(CONFIG["EVAL_EPISODES"]), desc=f"Eval {agent_name}"):
            state, _ = eval_env.reset()
            stage_count = 1
            ep_reward, ep_cost, ep_final_wrs, ep_final_die, ep_final_comp, ep_final_qual = 0, 0, 0, 0, 0, 0
            current_pipeline = []
            
            while True:
                # Use SOFTMAX for Q-Agents to get variety, standard sample for Baselines
                if isinstance(agent, ChatbotContextAwareQPlanner):
                    action = agent.sample_action_softmax(state, temperature=CONFIG["EVAL_TEMPERATURE"])
                else:
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
                    "Agent": agent_name,
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
            
            # Store history
            hist_reward.append(ep_reward)
            hist_wrs.append(ep_final_wrs)
            hist_die.append(ep_final_die)
            hist_comp.append(ep_final_comp)

            eval_rewards.append(ep_reward)
            eval_costs.append(ep_cost)
            eval_wrs.append(ep_final_wrs)
            eval_die.append(ep_final_die)
            eval_comp.append(ep_final_comp)
            eval_qual.append(ep_final_qual)
            pipeline_choices.append(tuple(current_pipeline))
        
        # Store in Eval Plot Data
        eval_plot_data[agent_name] = {
            "Reward": hist_reward, "WRS": hist_wrs, "DIE": hist_die, "Comp": hist_comp, "Color": color
        }

        write_header = not os.path.exists(CONFIG["LOG_FILE"]) or os.path.getsize(CONFIG["LOG_FILE"]) == 0
        df_chunk = pd.DataFrame(current_agent_trace)
        if not df_chunk.empty:
            df_chunk.to_csv(CONFIG["LOG_FILE"], mode='a', header=write_header, index=False)

        top3_str = "N/A"
        if pipeline_choices:
            counts = Counter(pipeline_choices).most_common(3)
            parts = []
            for pipe, count in counts:
                pipe_str = " -> ".join(pipe)
                parts.append(f"{pipe_str} ({count})")
            top3_str = "; ".join(parts)

        results_table.append({
            "Agent": agent_name,
            "Train Mode": "N/A", # Simplified for summary
            "Avg Reward": np.mean(eval_rewards),
            "Avg Cost": np.mean(eval_costs),
            "Avg WRS": np.mean(eval_wrs),
            "Avg DIE": np.mean(eval_die),
            "Avg Quality": np.mean(eval_qual),
            "Avg Compression": np.mean(eval_comp),
            "Top Pipeline": top3_str
        })

    # 4. Output Table
    df = pd.DataFrame(results_table)
    print("\n=== FINAL CHATBOT RESULTS ===")
    print(df[["Agent", "Avg Reward", "Avg Quality", "Avg Compression", "Avg WRS", "Avg DIE", "Top Pipeline"]].to_string(index=False))
    df.to_csv("chatbot_results_summary.csv", index=False)
    
    # 5. PLOTTING FUNCTION (Generic for both phases)
    def plot_metrics(data_dict, filename, title_prefix):
        if not data_dict: return
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))
        
        def smooth_data(d, window=10):
            if len(d) > window: return np.convolve(d, np.ones(window)/window, mode='valid')
            return d

        for name, d in data_dict.items():
            axes[0,0].plot(smooth_data(d["Reward"]), label=name, color=d["Color"])
        axes[0,0].set_title(f"{title_prefix} Total Reward")
        axes[0,0].legend()

        for name, d in data_dict.items():
            axes[0,1].plot(smooth_data(d["WRS"]), label=name, color=d["Color"])
        axes[0,1].set_title(f"{title_prefix} WRS Metric")

        for name, d in data_dict.items():
            axes[1,0].plot(smooth_data(d["DIE"]), label=name, color=d["Color"])
        axes[1,0].set_title(f"{title_prefix} DIE Metric")
        
        has_comp = False
        for name, d in data_dict.items():
            if "Comp" in d and d["Comp"]:
                axes[1,1].plot(smooth_data(d["Comp"]), label=name, color=d["Color"])
                has_comp = True
        if has_comp: axes[1,1].set_title(f"{title_prefix} Compression Reward")
        else: axes[1,1].set_visible(False)

        plt.tight_layout()
        plt.savefig(filename)
        print(f"\nSaved plots to '{filename}'")

    plot_metrics(plot_data, "chatbot_training_metrics.png", "Training")
    plot_metrics(eval_plot_data, "chatbot_evaluation_metrics.png", "Evaluation")

if __name__ == "__main__":
    run_experiment_full_trace()