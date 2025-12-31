import os
import sys

# CRITICAL FIX: Set HF_HOME to a local directory to avoid Permission Errors
# This forces the model to download to a folder we know we can write to.
os.environ['HF_HOME'] = './local_model_cache'

import pandas as pd
from transformers import pipeline
import torch
from tqdm import tqdm

# Try to import the extractive summarizer library
try:
    from summarizer import Summarizer
except ImportError:
    print("\n[ERROR] The 'bert-extractive-summarizer' library is missing.")
    print("Please install it using the following command:")
    print("pip install bert-extractive-summarizer")
    sys.exit(1)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
INPUT_FILE = 'data/responses/s1_no_sum_responses.csv'
OUTPUT_FILE = 'data/responses/s1_sum_responses.csv'

# List of columns to summarize
TARGET_COLUMNS = [
    'original_m_safechat', 
    'rt_es_m_safechat', 
    'rt_da_safechat'
]

# Parameters for summarization
MIN_LENGTH = 20       # Minimum words in abstractive summary
MAX_LENGTH = 60       # Maximum words in abstractive summary
MIN_INPUT_LENGTH = 30 # If text is shorter than this, we won't summarize it

def main():
    # 1. Load Data
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found.")
        return

    df = pd.read_csv(INPUT_FILE)
    print(f"Loaded {len(df)} rows from {INPUT_FILE}")

    # ---------------------------------------------------------
    # 2. LOAD MODELS
    # ---------------------------------------------------------
    print("\nLoading models... (This may take a moment)")
    
    # Ensure local cache directory exists
    if not os.path.exists('./local_model_cache'):
        os.makedirs('./local_model_cache')
    
    # Check device (GPU is much faster)
    device = 0 if torch.cuda.is_available() else -1
    print(f"Using device: {'GPU' if device == 0 else 'CPU'}")
    
    # A. Abstractive Model (BART)
    try:
        print("Loading Abstractive Model (BART)...")
        abstractive_pipe = pipeline(
            "summarization", 
            model="facebook/bart-large-cnn", 
            device=device
        )
    except Exception as e:
        print(f"Error loading abstractive model: {e}")
        return

    # B. Extractive Model (BERT)
    try:
        print("Loading Extractive Model (BERT)...")
        extractive_model = Summarizer()
    except Exception as e:
        print(f"Error loading extractive model: {e}")
        return

    # ---------------------------------------------------------
    # 3. GENERATE SUMMARIES FOR EACH COLUMN
    # ---------------------------------------------------------
    
    for col in TARGET_COLUMNS:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found. Skipping.")
            continue
            
        print(f"\nProcessing column: '{col}'...")
        
        # Ensure string type
        df[col] = df[col].fillna("").astype(str)
        
        abstractive_results = []
        extractive_results = []

        # Iterate with progress bar
        for text in tqdm(df[col], desc=f"Summarizing {col}"):
            word_count = len(text.split())

            # Skip summarization if text is too short or empty
            if word_count < MIN_INPUT_LENGTH:
                abstractive_results.append(text)
                extractive_results.append(text)
                continue

            # --- Abstractive ---
            try:
                # Generate summary. truncating ensures we don't crash on huge texts
                summary_abs = abstractive_pipe(
                    text, 
                    max_length=MAX_LENGTH, 
                    min_length=MIN_LENGTH, 
                    do_sample=False, 
                    truncation=True
                )
                abstractive_results.append(summary_abs[0]['summary_text'])
            except Exception as e:
                # Fallback in case of model error, keep original
                abstractive_results.append(text)

            # --- Extractive ---
            try:
                # ratio=0.5 means keep top 50% of sentences
                summary_ext = extractive_model(text, ratio=0.5) 
                # If BERT returns empty string (sometimes happens on very short text), use original
                if not summary_ext:
                    summary_ext = text
                extractive_results.append(summary_ext)
            except Exception as e:
                extractive_results.append(text)

        # Save results to new columns
        df[f'{col}_abstractive'] = abstractive_results
        df[f'{col}_extractive'] = extractive_results

    # ---------------------------------------------------------
    # 4. SAVE RESULTS
    # ---------------------------------------------------------
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Saved all summaries to: {OUTPUT_FILE}")
    
    # Preview
    print("\n--- Preview (Row 0) ---")
    first_col = TARGET_COLUMNS[0]
    if f'{first_col}_abstractive' in df.columns:
        print(f"ORIGINAL ({first_col}): {df[first_col].iloc[0][:100]}...")
        print(f"ABSTRACTIVE: {df[f'{first_col}_abstractive'].iloc[0]}")
        print(f"EXTRACTIVE: {df[f'{first_col}_extractive'].iloc[0]}")

if __name__ == "__main__":
    main()