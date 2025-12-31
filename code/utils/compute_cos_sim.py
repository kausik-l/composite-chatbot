import os
# CRITICAL FIX: Set HF_HOME before other imports to force all caching to a local directory
# This bypasses the "Permission denied" errors in the system cache.
os.environ['HF_HOME'] = './local_model_cache'

import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
RESPONSES_FILE = 'data/responses/s2_no_sum_responses.csv'
MASTER_FILE = 'data/processed/processed_hiv.csv'
OUTPUT_FILE = 'data/outcome/s2_no_sum.csv'

def main():
    print("1. Loading Datasets...")
    try:
        df_resp = pd.read_csv(RESPONSES_FILE)
        df_master = pd.read_csv(MASTER_FILE)
        print(f"   Responses loaded: {len(df_resp)}")
        print(f"   Master data loaded: {len(df_master)}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    # ---------------------------------------------------------
    # 2. ALIGN GROUND TRUTH
    # ---------------------------------------------------------
    print("2. Mapping Ground Truth Answers...")
    
    # We merge based on the prompt text to find the correct answer for each row
    # responses.csv uses 'prompt_original'
    # processed_hiv.csv uses 'prompt_text'
    
    # Ensure strings are clean
    df_resp['prompt_original'] = df_resp['prompt_original'].astype(str).str.strip()
    df_master['prompt_text'] = df_master['prompt_text'].astype(str).str.strip()
    
    # Merge 'original_answer' into the response dataframe
    df_merged = pd.merge(
        df_resp,
        df_master[['prompt_text', 'original_answer']],
        left_on='prompt_original',
        right_on='prompt_text',
        how='left'
    )
    
    # Validation
    missing_answers = df_merged['original_answer'].isnull().sum()
    if missing_answers > 0:
        print(f"   Warning: {missing_answers} rows match no ground truth answer. Scores will be 0.0.")
    
    df_merged['original_answer'] = df_merged['original_answer'].fillna("").astype(str)

    # ---------------------------------------------------------
    # 3. ENCODE & SCORE
    # ---------------------------------------------------------
    print(f"3. Loading Model ({MODEL_NAME})...")
    # Check for MPS (Mac M1/M2) or CUDA, else CPU
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"   Using device: {device}")
    
    # Ensure local cache directory exists
    local_cache = "./local_model_cache"
    os.makedirs(local_cache, exist_ok=True)
    
    print(f"   Loading model from/to cache: {local_cache}")
    try:
        # We rely on HF_HOME and cache_folder to handle downloads. 
        # force_download is not supported in __init__.
        model = SentenceTransformer(MODEL_NAME, device=device, cache_folder=local_cache)
    except Exception as e:
        print(f"   Error loading model: {e}")
        print("   Try deleting the './local_model_cache' folder and running again.")
        return

    # Dictionary mapping input column -> output score column
    cols_to_score = {
        'original_m_safechat': 'original_m_safechat_outcome',
        'rt_es_m_safechat':    'rt_es_m_safechat_outcome',
        'rt_da_m_safechat':    'rt_da_m_safechat_outcome'
    }

    # A. Encode Ground Truth (Reference)
    print("   Encoding Ground Truth Answers...")
    gt_embeddings = model.encode(
        df_merged['original_answer'].tolist(),
        convert_to_tensor=True,
        show_progress_bar=True,
        batch_size=64
    )

    # B. Encode & Compare Responses
    final_scores = {}

    for input_col, output_col in cols_to_score.items():
        print(f"   Processing {input_col} -> {output_col}...")
        
        if input_col not in df_merged.columns:
            print(f"   Error: Column '{input_col}' not found in CSV. Skipping.")
            continue
            
        # Clean response text
        response_texts = df_merged[input_col].fillna("").astype(str).tolist()
        
        # Encode
        resp_embeddings = model.encode(
            response_texts,
            convert_to_tensor=True,
            show_progress_bar=True,
            batch_size=64
        )
        
        # Compute Pairwise Cosine Similarity
        # This compares Row 1 of GT with Row 1 of Response, Row 2 with Row 2, etc.
        scores = util.pairwise_cos_sim(gt_embeddings, resp_embeddings)
        
        # Move to CPU and convert to numpy array for saving
        final_scores[output_col] = scores.cpu().numpy()

    # ---------------------------------------------------------
    # 4. SAVE FINAL OUTPUT
    # ---------------------------------------------------------
    print("4. Saving Data...")
    
    # Add scores to the dataframe
    for col_name, score_data in final_scores.items():
        df_merged[col_name] = score_data

    # Define the strict list of columns you requested
    final_columns_list = [
        'T', 
        'Z_race', 
        'Z_gender', 
        'original_m_safechat_outcome', 
        'rt_es_m_safechat_outcome', 
        'rt_da_m_safechat_outcome'
    ]
    
    # Verify columns exist before slicing
    available_cols = [c for c in final_columns_list if c in df_merged.columns]
    
    final_df = df_merged[available_cols]
    
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"Success! Saved processed file to: {OUTPUT_FILE}")
    print("\nPreview:")
    print(final_df.head())

if __name__ == "__main__":
    main()