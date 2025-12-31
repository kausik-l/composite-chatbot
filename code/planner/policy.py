import numpy as np
import pandas as pd
import random
import pickle
import os

class ChatbotContextAwareQPlanner:
    """
    Intelligent Agent: Context-Aware Q-Learning for Chatbot Pipeline.
    Tracks state context (Current Stage) to learn optimal paths.
    Includes logging for analysis.
    """
    def __init__(self, action_space, stage_map, alpha=0.1, gamma=0.99, epsilon=0.1, action_name="select_component"):
        self.q_table = {} 
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon  
        
        self.action_space = action_space
        self.stage_map = stage_map
        self.action_name = action_name 
        
        self.training_history = [] 
        self.episode_log = [] # List to store per-episode metrics

    def _parse_rddl_state(self, state):
        curr_stage = None
        # In this domain, we only track current_stage
        for key, val in state.items():
            if (val == True or val == 1) and "current_stage" in key:
                curr_stage = key.split("___")[-1]
        return curr_stage

    def get_state_key(self, state):
        s = self._parse_rddl_state(state)
        if not s: return "DONE"
        return f"{s}" # Key is just the stage (s1, s2, s3)

    def sample_action(self, state):
        state_key = self.get_state_key(state)
        
        # Check available actions for this stage
        curr_stage = self._parse_rddl_state(state)
        if not curr_stage or curr_stage not in self.stage_map:
            return {} # No valid actions or done

        available_actions = self.stage_map[curr_stage]
        
        # Epsilon-Greedy
        if random.random() < self.epsilon:
            chosen = random.choice(available_actions)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in available_actions}
            
            # Pick max Q
            q_vals = self.q_table[state_key]
            max_q = max(q_vals.values())
            # Tie breaking
            best_actions = [a for a, q in q_vals.items() if q == max_q]
            chosen = random.choice(best_actions)
        
        return {f"{self.action_name}___{chosen}": 1}

    def update(self, state, action, reward, next_state):
        curr_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        
        # Extract action name
        used_action = None
        for k, v in action.items():
            if v == 1: 
                used_action = k.split("___")[-1]
                break
        
        if curr_key != "DONE" and used_action:
            # Init Q-table if new
            if curr_key not in self.q_table:
                curr_stage = self._parse_rddl_state(state)
                if curr_stage in self.stage_map:
                    self.q_table[curr_key] = {a: 0.0 for a in self.stage_map[curr_stage]}
            
            if curr_key in self.q_table and used_action in self.q_table[curr_key]:
                old_q = self.q_table[curr_key][used_action]
                
                next_max_q = 0.0
                if next_key != "DONE":
                    if next_key not in self.q_table:
                        n_stage = self._parse_rddl_state(next_state)
                        if n_stage and n_stage in self.stage_map:
                            self.q_table[next_key] = {a: 0.0 for a in self.stage_map[n_stage]}
                    
                    if next_key in self.q_table:
                        next_max_q = max(self.q_table[next_key].values())

                new_q = old_q + self.alpha * (reward + self.gamma * next_max_q - old_q)
                self.q_table[curr_key][used_action] = new_q

    def log_episode(self, episode_num, mode, reward, wrs, die, cost, pipeline_str):
        """
        Logs metrics for a single episode.
        """
        self.episode_log.append({
            "Episode": episode_num,
            "Mode": mode,
            "Total Reward": reward,
            "WRS": wrs,
            "DIE": die,
            "Cost": cost,
            "Pipeline": pipeline_str
        })

    def save_log(self, filepath="chatbot_episode_log.csv"):
        """
        Saves the accumulated log to a CSV file.
        """
        if not self.episode_log:
            print("No logs to save.")
            return
        
        df = pd.DataFrame(self.episode_log)
        df.to_csv(filepath, index=False)
        print(f"Episode log saved to {filepath}")

    def save_agent(self, filepath="chatbot_q_agent.pkl"):
        data = {
            "q_table": self.q_table,
            "hyperparams": {"alpha": self.alpha, "gamma": self.gamma, "epsilon": self.epsilon}
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"Agent saved to {filepath}")