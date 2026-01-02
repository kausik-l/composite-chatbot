import pandas as pd
from transformers import pipeline
from summarizer import Summarizer
import torch
from tqdm import tqdm
import os
import sys
import gc

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
# CRITICAL FIX: Set HF_HOME to a local directory to avoid Permission Errors
os.environ['HF_HOME'] = './local_model_cache'

# Input: The file with raw responses (e.g., s1_no_sum.csv)
INPUT_FILE = 'data/responses/s1_no_sum_responses.csv' 

# Output Prefixes for two separate files
OUTPUT_FILE_ABS = 'data/responses/s1_sum1_responses.csv'
OUTPUT_FILE_EXT = 'data/responses/s1_sum2_responses.csv'

# Columns to summarize
TARGET_COLUMNS = [
    'original_m_safechat', 
    'rt_es_m_safechat', 
    'rt_da_m_safechat'
]

# Summarization Params
# We switch to a lighter model to avoid Bus Error 10
MODEL_NAME = "sshleifer/distilbart-cnn-12-6" 
BATCH_SIZE = 8 
MIN_LENGTH = 15
MAX_LENGTH = 50

def load_models():
    print("Loading models...")
    if not os.path.exists('./local_model_cache'):
        os.makedirs('./local_model_cache')
        
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    # Abstractive (DistilBART)
    try:
        abstractive_pipe = pipeline(
            "summarization", 
            model=MODEL_NAME, 
            device=device,
            batch_size=BATCH_SIZE
        )
    except Exception as e:
        print(f"Error loading Abstractive Model: {e}")
        return None, None

    # Extractive (BERT)
    try:
        extractive_model = Summarizer()
    except Exception as e:
        print(f"Error loading BERT Summarizer: {e}")
        return None, None
        
    return abstractive_pipe, extractive_model

def summarize_batch_abstractive(pipe, texts):
    """Summarizes a list of texts using the pipeline."""
    try:
        valid_inputs = []
        indices = []
        results = [""] * len(texts)
        
        for i, text in enumerate(texts):
            # Only summarize if text is reasonably long
            if len(str(text).split()) > 10: 
                valid_inputs.append(str(text))
                indices.append(i)
            else:
                results[i] = str(text) 

        if not valid_inputs:
            return results

        # Run batch inference
        summaries = pipe(
            valid_inputs, 
            max_length=MAX_LENGTH, 
            min_length=MIN_LENGTH, 
            do_sample=False, 
            truncation=True
        )
        
        for idx, summary in zip(indices, summaries):
            results[idx] = summary['summary_text']
            
        return results
    except Exception as e:
        print(f"Batch Error: {e}")
        return [str(t) for t in texts]

def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows.")

    abs_pipe, ext_model = load_models()
    if not abs_pipe: return

    df_abs = df.copy()
    df_ext = df.copy()

    for col in TARGET_COLUMNS:
        if col not in df.columns:
            print(f"Skipping {col} (not found)")
            continue
            
        print(f"\nProcessing {col}...")
        original_texts = df[col].fillna("").astype(str).tolist()
        
        # 1. Abstractive
        print("  > Generating Abstractive Summaries...")
        abs_summaries = []
        for i in tqdm(range(0, len(original_texts), BATCH_SIZE)):
            batch_texts = original_texts[i : i+BATCH_SIZE]
            batch_sums = summarize_batch_abstractive(abs_pipe, batch_texts)
            abs_summaries.extend(batch_sums)
            
            # AGGRESSIVE MEMORY CLEANUP
            if i % (BATCH_SIZE * 10) == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        
        # Replace column in df_abs
        df_abs[col] = abs_summaries

        # 2. Extractive
        print("  > Generating Extractive Summaries...")
        ext_summaries = []
        for text in tqdm(original_texts):
            if len(text.split()) > 10:
                try:
                    summary = ext_model(text, ratio=0.5)
                    ext_summaries.append(summary if summary else text)
                except:
                    ext_summaries.append(text)
            else:
                ext_summaries.append(text)
        
        # Replace column in df_ext
        df_ext[col] = ext_summaries

    # Save Files
    df_abs.to_csv(OUTPUT_FILE_ABS, index=False)
    print(f"\nSuccess! Abstractive summaries saved to {OUTPUT_FILE_ABS}")
    
    df_ext.to_csv(OUTPUT_FILE_EXT, index=False)
    print(f"Success! Extractive summaries saved to {OUTPUT_FILE_EXT}")

if __name__ == "__main__":
    main()