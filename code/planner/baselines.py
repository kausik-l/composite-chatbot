import random
import numpy as np
import os
import pandas as pd
from env.metric_utils import calc_wrs

class ChatbotRandomPipelinePlanner:
    def __init__(self, stage_map, action_name="select_component"):
        self.stage_map = stage_map
        self.action_name = action_name

    def sample_action(self, state):
        current_stage = self._get_stage(state)
        if current_stage and current_stage in self.stage_map:
            choice = random.choice(self.stage_map[current_stage])
            return {f"{self.action_name}___{choice}": 1}
        return {} 

    def _get_stage(self, state):
        for k, v in state.items():
            if "current_stage" in k and (v == True or v == 1):
                return k.split("___")[-1]
        return None
    
    def update(self, s, a, r, ns): pass

class ChatbotFixedPipelinePlanner:
    def __init__(self, stage_map, selection_index=0, name="Fixed", action_name="select_component"):
        self.stage_map = stage_map
        self.selection_index = selection_index
        self.name = name
        self.action_name = action_name

    def sample_action(self, state):
        current_stage = self._get_stage(state)
        if current_stage and current_stage in self.stage_map:
            options = self.stage_map[current_stage]
            idx = min(self.selection_index, len(options) - 1)
            choice = options[idx]
            return {f"{self.action_name}___{choice}": 1}
        return {}

    def _get_stage(self, state):
        for k, v in state.items():
            if "current_stage" in k and (v == True or v == 1):
                return k.split("___")[-1]
        return None

    def update(self, s, a, r, ns): pass

class ChatbotLookaheadFairnessPlanner:
    """
    Heuristic: Greedy Lookahead for WRS.
    It simulates each valid action, checks the resulting WRS (if possible), and picks the best.
    Note: Real lookahead requires access to the Env to know 'current_system' etc.
    """
    def __init__(self, stage_map, env, action_name="select_component"):
        self.stage_map = stage_map
        self.env = env
        self.action_name = action_name

    def sample_action(self, state):
        current_stage = self._get_stage(state)
        if not current_stage or current_stage not in self.stage_map:
            return {}

        options = self.stage_map[current_stage]
        best_action = None
        min_wrs = float('inf')
        
        # If we are at the final decision stage (s3 - Summarization), we can peek at files.
        # At s1/s2, we can't calculate WRS yet because the pipeline isn't finished.
        # So at s1/s2, we behave Randomly (or use a heuristic if we had one).
        if current_stage == 's3': 
            # We assume env has current_system set
            sys_num = self.env.current_system.replace("sys_s", "") if self.env.current_system else "1"
            
            # Determine target column based on current_paraphrase
            # Default to original if not set
            target_col = "original_outcome"
            if self.env.current_paraphrase == "para_spanish": target_col = "rt_es_outcome"
            elif self.env.current_paraphrase == "para_danish": target_col = "rt_da_outcome"

            for opt in options:
                # opt is sum_sum1, sum_sum2, sum_no
                if "sum1" in opt: suffix = "sum1"
                elif "sum2" in opt: suffix = "sum2"
                else: suffix = "no_sum"
                
                fname = f"s{sys_num}_{suffix}_responses.csv"
                fpath = os.path.join(self.env.data_dir, fname)
                # Fallback
                if not os.path.exists(fpath):
                    fpath = os.path.join(self.env.data_dir, f"s{sys_num}_{suffix}.csv")
                
                current_wrs = 1.0 # High default
                
                if os.path.exists(fpath):
                    try:
                        # Load a small sample to be fast
                        df = pd.read_csv(fpath) # In production, maybe read just header+sample
                        if target_col in df.columns:
                            # Calc WRS for Race
                            wrs_race = calc_wrs(df, 'Z_race', target_col)
                            # Calc WRS for Gender (if we want robust lookahead)
                            wrs_gender = calc_wrs(df, 'Z_gender', target_col)
                            current_wrs = wrs_race + wrs_gender
                    except:
                        pass
                
                if current_wrs < min_wrs:
                    min_wrs = current_wrs
                    best_action = opt
            
            if best_action:
                return {f"{self.action_name}___{best_action}": 1}
        
        # Fallback (Random) for non-final stages or failures
        choice = random.choice(options)
        return {f"{self.action_name}___{choice}": 1}

    def _get_stage(self, state):
        for k, v in state.items():
            if "current_stage" in k and (v == True or v == 1):
                return k.split("___")[-1]
        return None

    def update(self, s, a, r, ns): pass