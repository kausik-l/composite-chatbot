import pandas as pd
import os
import numpy as np

# Configuration
DATA_DIR = "data/responses" 


def calculate_compression_scores():
    print("Calculating compression scores (Summary vs No-Summary Baseline)...")
    
    systems = ['s1', 's2', 's3']
    modes = ['sum1', 'sum2'] # We only calculate FOR summary modes
    
    # Column Map (Score Col -> Text Col)
    # Adjust based on actual column names in your _responses.csv files
    # Assuming the text columns are named 'original_m_safechat', 'rt_es_m_safechat', etc.
    cols_map = {
        "original_m_safechat_outcome": "original_m_safechat",
        "rt_es_m_safechat_outcome": "rt_es_m_safechat",
        "rt_da_m_safechat_outcome": "rt_da_m_safechat" 
    }

    def get_word_count(text):
        return len(str(text).split())

    for sys_name in systems:
        # 1. Load Baseline (No Sum)
        # Try finding the no_sum responses file
        baseline_candidates = [
            f"{sys_name}_no_sum_responses.csv" 
        ]
        
        baseline_path = None
        for f in baseline_candidates:
            if os.path.exists(os.path.join(DATA_DIR, f)):
                baseline_path = os.path.join(DATA_DIR, f)
                break
        
        if not baseline_path:
            print(f"Skipping {sys_name}: Baseline (no_sum) file not found.")
            continue
            
        print(f"Loaded Baseline for {sys_name}: {os.path.basename(baseline_path)}")
        df_base = pd.read_csv(baseline_path)

        # 2. Process Variants (sum1, sum2)
        for mode in modes:
            # Try finding the target response file
            target_candidates = [
                f"{sys_name}_{mode}_responses.csv",
                f"{sys_name}_{mode}.csv"
            ]
            
            target_path = None
            target_filename = None
            for f in target_candidates:
                if os.path.exists(os.path.join(DATA_DIR, f)):
                    target_path = os.path.join(DATA_DIR, f)
                    target_filename = f
                    break
            
            if not target_path:
                continue
                
            print(f"  Processing {target_filename}...")
            df_target = pd.read_csv(target_path)
            
            # Ensure alignment (truncate to min length)
            min_len = min(len(df_base), len(df_target))
            
            for score_col, text_col in cols_map.items():
                comp_col_name = score_col.replace("_outcome", "_compression")
                
                # Check if text column exists in both
                if text_col in df_base.columns and text_col in df_target.columns:
                    # Calculate Lengths
                    len_base = df_base[text_col].iloc[:min_len].apply(get_word_count)
                    len_target = df_target[text_col].iloc[:min_len].apply(get_word_count)
                    
                    # Avoid zero division
                    len_base = len_base.replace(0, 1)
                    
                    # Ratio: 1 - (Target / Baseline)
                    # Positive = Compressed. Negative = Expanded.
                    ratio = 1.0 - (len_target / len_base)
                    
                    ratio = ratio.clip(-0.5, 1.0) 
                    
                    # Create/Overwrite column with 0s first
                    values = np.zeros(len(df_target))
                    values[:min_len] = ratio.values
                    df_target[comp_col_name] = values
                else:
                    # If text column missing, set to 0
                    if text_col not in df_target.columns:
                        # Maybe we are processing a score-only file but trying to match it to text?
                        # If so, we can't calculate.
                        pass
                    df_target[comp_col_name] = 0.0

            df_target.to_csv(target_path, index=False)
            print(f"    -> Updated compression columns.")

    # Also update 'no_sum' files to have 0.0 compression columns for consistency
    for sys_name in systems:
        f = f"{sys_name}_no_sum_responses.csv" # Or .csv
        # We need to find the actual file being used
        candidates = [f"{sys_name}_no_sum_responses.csv", f"{sys_name}_no_sum.csv"]
        for fname in candidates:
            fpath = os.path.join(DATA_DIR, fname)
            if os.path.exists(fpath):
                df = pd.read_csv(fpath)
                updated = False
                for score_col in cols_map.keys():
                    comp_col = score_col.replace("_outcome", "_compression")
                    if comp_col not in df.columns: # Or overwrite
                        df[comp_col] = 0.0
                        updated = True
                    else:
                        # Reset to 0 just in case
                        df[comp_col] = 0.0
                        updated = True
                if updated:
                    df.to_csv(fpath, index=False)
                    print(f"  -> Reset 0.0 columns for {fname}")

if __name__ == "__main__":
    calculate_compression_scores()