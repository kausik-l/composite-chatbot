import pandas as pd
import numpy as np
import os
from pyRDDLGym.core.env import RDDLEnv
from env.metric_utils import calc_wrs
from utils.causal_metrics import compute_arc_metrics

class ChatbotPipelineEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_dir, batch_size=100, reward_mode="WRS"):
        super().__init__(domain=domain_file, instance=instance_file)
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.reward_mode = reward_mode
        self.sampled_batch = None
        
        # Track selections to load the correct file/column at the end
        self.current_paraphrase = None 
        self.current_system = None
        self.current_summary = None

    def reset(self):
        obs, _ = super().reset()
        self.current_paraphrase = None
        self.current_system = None
        self.current_summary = None
        self.sampled_batch = None
        return obs, {}

    def step(self, action):
        obs, rddl_reward, terminated, truncated, info = super().step(action)
        done = terminated or truncated
        
        # Identify Action from RDDL action dict
        selected_comp = None
        for k, v in action.items():
            if v == 1 and "select_component" in k:
                selected_comp = k.split("___")[-1]
                break
        
        if selected_comp:
            # Stage 1: Paraphrase
            if "para_" in selected_comp:
                self.current_paraphrase = selected_comp # e.g., para_none, para_spanish
            
            # Stage 2: System
            elif "sys_" in selected_comp:
                self.current_system = selected_comp # e.g., sys_s1
            
            # Stage 3: Summary
            elif "sum_" in selected_comp:
                self.current_summary = selected_comp # e.g., sum_yes, sum_no

        fairness_penalty = 0.0
        raw_wrs = 0.0
        raw_die = 0.0

        # CALCULATE REWARD ONLY AT THE END OF THE PIPELINE
        if done and self.current_paraphrase and self.current_system and self.current_summary:
            
            # 1. Determine which file to load
            # Format: s{1,2,3}_{sum,no_sum}.csv
            # sys_s1 -> "1"
            sys_num = self.current_system.replace("sys_s", "") 
            # sum_yes -> "sum", sum_no -> "no_sum"
            sum_suffix = "sum" if self.current_summary == "sum_yes" else "no_sum"
            
            filename = f"s{sys_num}_{sum_suffix}.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            if os.path.exists(filepath):
                # Load the specific dataset for this system configuration
                full_df = pd.read_csv(filepath)
                
                # Sample a batch
                n_sample = min(self.batch_size, len(full_df))
                self.sampled_batch = full_df.sample(n=n_sample).copy()
                
                # 2. Determine which column to use based on Paraphrase choice
                # para_none -> original_m_safechat_outcome
                # para_spanish -> rt_es_m_safechat_outcome
                # para_danish -> rt_da_m_safechat_outcome
                
                target_col = "original_m_safechat_outcome" # default
                if self.current_paraphrase == "para_spanish":
                    target_col = "rt_es_m_safechat_outcome"
                elif self.current_paraphrase == "para_danish":
                    target_col = "rt_da_m_safechat_outcome"
                
                # 3. Calculate Metrics (WRS / DIE)
                # Protected Attribute: Z_race (1=AA, 2=EA, 0=None)
                
                if target_col in self.sampled_batch.columns:
                    # WRS (Weighted Rejection Score)
                    try:
                        # We use Z_race. metric_utils treats >0 as "protected group"
                        # Ideally we compare AA (1) vs Baseline (0) or AA (1) vs EA (2)
                        # The generic calc_wrs checks abs(mean(Y|Z=1) - mean(Y|Z=0))
                        # Here, Z=0 is Baseline (Anonymous). Z=1 is AA.
                        wrs_race = calc_wrs(self.sampled_batch, 'Z_race', target_col)
                        raw_wrs = wrs_race
                    except: 
                        raw_wrs = 0.0

                    # DIE (Causal Metric)
                    # We need a treatment column T (which exists in your CSV)
                    try:
                        # T=0 (Anonymous), T=1 (Identity Revealed)
                        metrics = compute_arc_metrics(
                            self.sampled_batch, 
                            treatment_col='T',       
                            outcome_col=target_col, 
                            confounders=['Z_race', 'Z_gender']
                        )
                        raw_die = abs(metrics['DIE_Confounding'])
                    except: 
                        raw_die = 0.0

                    # 4. Apply Penalty
                    if self.reward_mode == "WRS":
                        fairness_penalty += (raw_wrs * 10.0)
                    elif self.reward_mode == "DIE":
                        fairness_penalty += (raw_die * 10.0)
                    elif self.reward_mode == "BOTH":
                        fairness_penalty += (raw_wrs * 5.0) + (raw_die * 5.0)

            else:
                # File missing (e.g. s3_sum.csv not generated yet)
                # Penalty for invalid path, but allow simulation to continue
                fairness_penalty = 5.0 

        # Total Reward = (Negative Cost from RDDL) - Fairness Penalty
        total_reward = rddl_reward - fairness_penalty 
        
        if 'metrics' not in info: info['metrics'] = {}
        info['metrics']['fairness_penalty'] = fairness_penalty
        info['metrics']['rddl_cost'] = abs(rddl_reward)
        info['metrics']['raw_wrs'] = float(raw_wrs)
        info['metrics']['raw_die'] = float(raw_die)

        return obs, total_reward, done, truncated, info