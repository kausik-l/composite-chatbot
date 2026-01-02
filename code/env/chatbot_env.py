import pandas as pd
import numpy as np
import os
from pyRDDLGym.core.env import RDDLEnv
from env.metric_utils import calc_wrs
from utils.causal_metrics import compute_arc_metrics

class ChatbotPipelineEnv(RDDLEnv):
    def __init__(self, domain_file, instance_file, data_dir, batch_size=50, reward_mode="WRS"):
        super().__init__(domain=domain_file, instance=instance_file)
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.reward_mode = reward_mode
        self.sampled_batch = None
        
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
        
        selected_comp = None
        for k, v in action.items():
            if v == 1 and "select_component" in k:
                selected_comp = k.split("___")[-1]
                break
        
        if selected_comp:
            if "para_" in selected_comp: self.current_paraphrase = selected_comp
            elif "sys_" in selected_comp: self.current_system = selected_comp
            elif "sum_" in selected_comp: self.current_summary = selected_comp

        fairness_penalty = 0.0
        compression_reward = 0.0
        quality_reward = 0.0
        raw_wrs = 0.0
        raw_die = 0.0

        if done and self.current_paraphrase and self.current_system and self.current_summary:
            
            sys_num = self.current_system.replace("sys_s", "") 
            
            # Determine Summary Suffix based on selection
            if "sum1" in self.current_summary: sum_suffix = "sum1"
            elif "sum2" in self.current_summary: sum_suffix = "sum2"
            else: sum_suffix = "no_sum"
            
            # Construct filename: s{sys}_{suffix}_responses.csv
            filename = f"s{sys_num}_{sum_suffix}_responses.csv"
            filepath = os.path.join(self.data_dir, filename)
            
            # Fallback check
            if not os.path.exists(filepath):
                 filepath = os.path.join(self.data_dir, f"s{sys_num}_{sum_suffix}.csv")

            full_df = pd.read_csv(filepath)
            n_sample = min(self.batch_size, len(full_df))
            self.sampled_batch = full_df.sample(n=n_sample).copy()
            
            target_col = "original_outcome"
            if self.current_paraphrase == "para_spanish": target_col = "rt_es_outcome"
            elif self.current_paraphrase == "para_danish": target_col = "rt_da_outcome"
            
            comp_col = target_col.replace("_outcome", "_m_safechat_compression")

            # 1. QUALITY REWARD (Cosine Similarity)
            if target_col in self.sampled_batch.columns:
                avg_sim = self.sampled_batch[target_col].mean()
                quality_reward = avg_sim 

            # 2. COMPRESSION REWARD
            if comp_col in self.sampled_batch.columns:
                avg_comp = self.sampled_batch[comp_col].mean()
                compression_reward = avg_comp 
            
            # 3. FAIRNESS METRICS
            if target_col in self.sampled_batch.columns:
                # WRS: Summing Race AND Gender
                wrs_race = calc_wrs(self.sampled_batch, 'Z_race', target_col)
                
                wrs_gender = calc_wrs(self.sampled_batch, 'Z_gender', target_col)
                
                raw_wrs = (wrs_race + wrs_gender)/2

                # DIE
                try:
                    metrics = compute_arc_metrics(
                        self.sampled_batch, 
                        treatment_col='T',       
                        outcome_col=target_col, 
                        confounders=['Z_race', 'Z_gender']
                    )
                    raw_die = abs(metrics['DIE_Confounding'])
                except: raw_die = 0.0

                if self.reward_mode == "WRS": fairness_penalty += (raw_wrs * 100.0)
                elif self.reward_mode == "DIE": fairness_penalty += (raw_die * 100.0)
                elif self.reward_mode == "BOTH": fairness_penalty += (raw_wrs * 100.0) + (raw_die * 100.0)


        total_reward = rddl_reward + quality_reward + compression_reward - fairness_penalty
        
        if 'metrics' not in info: info['metrics'] = {}
        info['metrics']['fairness_penalty'] = fairness_penalty
        info['metrics']['compression_reward'] = compression_reward
        info['metrics']['quality_reward'] = quality_reward
        info['metrics']['rddl_cost'] = abs(rddl_reward) 
        info['metrics']['raw_wrs'] = float(raw_wrs)
        info['metrics']['raw_die'] = float(raw_die)

        return obs, total_reward, done, truncated, info