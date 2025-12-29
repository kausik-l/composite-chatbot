import pandas as pd
import requests
import concurrent.futures
import time
from tqdm import tqdm  # Progress bar


# S1 (LLM with RAG on UNAIDS queries) and S2 (SafeChat trained on UNAIDS).
API_URL = "http://13.59.191.223:8000/chat"
SYSTEM_PROMPT = "s2"  # You can change to "s2" if needed

# File Mappings
FILES = {
    'original': 'data/processed/processed_hiv.csv',
    'es': 'data/processed/processed_hiv_rt_es.csv',
    'da': 'data/processed/processed_hiv_rt_da.csv',
}

OUTPUT_FILE = "data/responses/responses.csv"


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def get_llm_response(prompt, system_id, retries=3):
    """
    Sends a request to the API with error handling and retries.
    """
    payload = {
        "message": prompt,
        "system": system_id
    }
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=20)
            if response.status_code == 200:
                # Adjust this key based on the actual API response structure
                # The user example showed: print(response.json())
                # usually it's ['response'] or similar. 
                # I will save the whole JSON string if unsure, or try to access 'response'.
                data = response.json()
                return data.get('response', str(data)) 
            else:
                time.sleep(1)
        except Exception as e:
            time.sleep(1)
            
    return "ERROR_TIMEOUT"

def process_row_tuple(row_data):
    """
    Worker function to process a single logical data point (one row across 3 files).
    """
    index, orig_row, es_row, da_row = row_data
    
    # 1. Get Responses for all 3 versions
    # We do this sequentially inside the thread to ensure they stay grouped, 
    # but the threads run in parallel.
    resp_original = get_llm_response(orig_row['prompt_text'], SYSTEM_PROMPT)
    resp_es = get_llm_response(es_row['prompt_text'], SYSTEM_PROMPT)
    resp_da = get_llm_response(da_row['prompt_text'], SYSTEM_PROMPT)
    
    # DEBUG: Print the first few results to console so user can verify
    if index < 3:
        print(f"\n--- [DEBUG PREVIEW] Row {index} ---")
        print(f"Original Q: {orig_row['prompt_text'][:60]}...")
        print(f"Original A: {resp_original[:60]}...")
        print(f"Spanish Q:  {es_row['prompt_text'][:60]}...")
        print(f"Spanish A:  {resp_es[:60]}...")
        print("-----------------------------------")

    # 2. Construct the Result Row
    # We keep the metadata from the original row (Race, Gender, T)
    return {
        'row_id': index,
        
        # Causal Metadata (Shared)
        'T': orig_row['T'],
        'Z_race': orig_row['Z_race'],
        'Z_gender': orig_row['Z_gender'],
        'category': orig_row['category'],
        # 'original_question': orig_row['original_question'],
        
        # The 3 Prompts (for reference)
        'prompt_original': orig_row['prompt_text'],
        'prompt_es': es_row['prompt_text'],
        'prompt_da': da_row['prompt_text'],
        
        # The 3 Responses (Requested Columns)
        'original_m_safechat': resp_original,
        'rt_es_m_safechat': resp_es,
        'rt_da_safechat': resp_da
    }

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def main():
    print("Loading datasets...")
    try:
        df_orig = pd.read_csv(FILES['original'])
        df_es = pd.read_csv(FILES['es'])
        df_da = pd.read_csv(FILES['da'])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure you run the preprocessing script first.")
        return

    # VALIDATION: Ensure all dataframes have the same length
    if not (len(df_orig) == len(df_es) == len(df_da)):
        print("Error: Datasets do not match in length. Cannot align rows safely.")
        return

    print(f"Loaded {len(df_orig)} rows per dataset.")
    print(f"Starting inference on System: {SYSTEM_PROMPT}")
    
    # Prepare data for threading
    # We zip them together so we process index 0 of all files, then index 1, etc.
    tasks = []
    for i in range(len(df_orig)):
        tasks.append((
            i, 
            df_orig.iloc[i], 
            df_es.iloc[i], 
            df_da.iloc[i]
        ))

    results = []
    
    # Run with a ThreadPool
    # Adjust max_workers based on how much load your API server can handle.
    # 10 is usually safe for a robust server.
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        # We use tqdm to show a progress bar
        results = list(tqdm(executor.map(process_row_tuple, tasks), total=len(tasks)))

    # Convert to DataFrame
    final_df = pd.DataFrame(results)
    
    # Sort by ID to ensure order is preserved
    final_df = final_df.sort_values('row_id').drop(columns=['row_id'])

    # Save
    final_df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSuccess! Consolidated results saved to {OUTPUT_FILE}")
    print(final_df[['T', 'Z_race', 'original_m_safechat', 'rt_es_m_safechat']].head())

if __name__ == "__main__":
    main()