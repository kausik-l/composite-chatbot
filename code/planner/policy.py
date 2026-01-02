import numpy as np
import random
import pickle
import os

class ChatbotContextAwareQPlanner:
    def __init__(self, action_space, stage_map, alpha=0.1, gamma=0.99, epsilon=0.1, action_name="select_component"):
        self.q_table = {} 
        self.alpha = alpha      
        self.gamma = gamma      
        self.epsilon = epsilon  
        self.action_space = action_space
        self.stage_map = stage_map
        self.action_name = action_name 

    def _parse_rddl_state(self, state):
        curr_stage = None
        for key, val in state.items():
            if (val == True or val == 1) and "current_stage" in key:
                curr_stage = key.split("___")[-1]
        return curr_stage

    def get_state_key(self, state):
        s = self._parse_rddl_state(state)
        if not s: return "DONE"
        return f"{s}"

    def sample_action(self, state):
        # Standard Epsilon-Greedy
        state_key = self.get_state_key(state)
        curr_stage = self._parse_rddl_state(state)
        
        if not curr_stage or curr_stage not in self.stage_map:
            return {} 

        available_actions = self.stage_map[curr_stage]
        
        if random.random() < self.epsilon:
            chosen = random.choice(available_actions)
        else:
            if state_key not in self.q_table:
                self.q_table[state_key] = {a: 0.0 for a in available_actions}
            q_vals = self.q_table[state_key]
            max_q = max(q_vals.values())
            best_actions = [a for a, q in q_vals.items() if q == max_q]
            chosen = random.choice(best_actions)
        
        return {f"{self.action_name}___{chosen}": 1}

    def sample_action_softmax(self, state, temperature=1.0):
        # Softmax selection for diversity during evaluation
        state_key = self.get_state_key(state)
        curr_stage = self._parse_rddl_state(state)
        
        if not curr_stage or curr_stage not in self.stage_map:
            return {} 

        available_actions = self.stage_map[curr_stage]
        
        if state_key not in self.q_table:
             # If unknown state, uniform random
             chosen = random.choice(available_actions)
        else:
            q_vals = self.q_table[state_key]
            # Extract Q-values in order
            actions = list(q_vals.keys())
            values = np.array([q_vals[a] for a in actions])
            
            # Softmax calculation
            # Subtract max for numerical stability
            exp_values = np.exp((values - np.max(values)) / temperature)
            probs = exp_values / np.sum(exp_values)
            
            chosen = np.random.choice(actions, p=probs)
            
        return {f"{self.action_name}___{chosen}": 1}

    def update(self, state, action, reward, next_state):
        curr_key = self.get_state_key(state)
        next_key = self.get_state_key(next_state)
        
        used_action = None
        for k, v in action.items():
            if v == 1: 
                used_action = k.split("___")[-1]
                break
        
        if curr_key != "DONE" and used_action:
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