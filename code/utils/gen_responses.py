import pandas as pd
import requests
import concurrent.futures
import time
import os
import csv
from tqdm import tqdm

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
API_URL = "http://13.59.191.223:8000/chat"
SYSTEM_PROMPT = "s1"
MAX_CONSECUTIVE_ERRORS = 5  # Stop script if this many fail in a row

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
    payload = {"message": prompt, "system": system_id}
    
    for attempt in range(retries):
        try:
            response = requests.post(API_URL, json=payload, timeout=45)
            if response.status_code == 200:
                data = response.json()
                return data.get('response', str(data))
            elif response.status_code >= 500:
                pass # Server error, retry
        except requests.exceptions.RequestException:
            pass # Network error, retry
            
        time.sleep(2 ** attempt) # Exponential backoff
            
    return "ERROR_TIMEOUT"

def process_row_tuple(row_data):
    index, orig_row, es_row, da_row = row_data
    
    # 1. Get Responses
    resp_original = get_llm_response(orig_row['prompt_text'], SYSTEM_PROMPT)
    resp_es = get_llm_response(es_row['prompt_text'], SYSTEM_PROMPT)
    resp_da = get_llm_response(da_row['prompt_text'], SYSTEM_PROMPT)
    
    # 2. Construct Result
    return {
        # 'row_id': index, 
        'T': orig_row['T'],
        'Z_race': orig_row['Z_race'],
        'Z_gender': orig_row['Z_gender'],
        'category': orig_row['category'],
        # 'original_question': orig_row['original_question'],
        'prompt_original': orig_row['prompt_text'],
        'prompt_es': es_row['prompt_text'],
        'prompt_da': da_row['prompt_text'],
        'original_m_safechat': resp_original,
        'rt_es_m_safechat': resp_es,
        'rt_da_safechat': resp_da
    }

# ---------------------------------------------------------
# MAIN EXECUTION
# ---------------------------------------------------------

def main():
    print("Loading input datasets...")
    try:
        df_orig = pd.read_csv(FILES['original'])
        df_es = pd.read_csv(FILES['es'])
        df_da = pd.read_csv(FILES['da'])
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return

    if not (len(df_orig) == len(df_es) == len(df_da)):
        print("Error: Datasets do not match in length.")
        return

    total_rows = len(df_orig)
    
    # -----------------------------------------------------
    # SMART RESUME LOGIC
    # -----------------------------------------------------
    completed_prompts = set()
    write_header = True
    
    if os.path.exists(OUTPUT_FILE):
        print(f"Scanning existing file: {OUTPUT_FILE}...")
        try:
            with open(OUTPUT_FILE, 'r', encoding='utf-8', errors='replace') as f:
                reader = csv.DictReader(f)
                if reader.fieldnames and 'prompt_original' in reader.fieldnames:
                    write_header = False 
                    for row in reader:
                        p_orig = row.get('prompt_original')
                        ans_orig = row.get('original_m_safechat')
                        
                        # Only mark as done if answer is valid (no error)
                        if p_orig and ans_orig and "ERROR_TIMEOUT" not in ans_orig:
                            completed_prompts.add(p_orig)
                else:
                    print("Warning: Headers missing or mismatch. Starting fresh/appending.")
                    write_header = False 
            print(f"Found {len(completed_prompts)} successfully completed rows.")
        except Exception as e:
            print(f"Could not read existing file ({e}). Starting fresh.")

    # -----------------------------------------------------
    # FILTER TASKS
    # -----------------------------------------------------
    tasks_to_run = []
    for i in range(total_rows):
        current_prompt = df_orig.iloc[i]['prompt_text']
        if current_prompt not in completed_prompts:
            tasks_to_run.append((i, df_orig.iloc[i], df_es.iloc[i], df_da.iloc[i]))
            
    if not tasks_to_run:
        print("All tasks are already completed!")
        return

    print(f"Starting inference on {len(tasks_to_run)} remaining rows...")

    # -----------------------------------------------------
    # EXECUTE WITH CIRCUIT BREAKER
    # -----------------------------------------------------
    fieldnames = [
        'T', 'Z_race', 'Z_gender', 'category', 'original_question',
        'prompt_original', 'prompt_es', 'prompt_da',
        'original_m_safechat', 'rt_es_m_safechat', 'rt_da_safechat'
    ]

    consecutive_errors = 0

    with open(OUTPUT_FILE, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # We treat results as an iterator
            results_generator = executor.map(process_row_tuple, tasks_to_run)
            
            # Using manual loop instead of tqdm helper to control flow
            pbar = tqdm(total=len(tasks_to_run))
            
            for result in results_generator:
                # Check for errors in this result
                responses = [
                    result['original_m_safechat'], 
                    result['rt_es_m_safechat'], 
                    result['rt_da_safechat']
                ]
                
                # If ANY of the 3 responses in this row failed
                if any("ERROR_TIMEOUT" in r for r in responses):
                    consecutive_errors += 1
                    tqdm.write(f"Row: [TIMEOUT] (Consecutive Errors: {consecutive_errors})")
                else:
                    consecutive_errors = 0 # Reset on success
                    # Optional: Print success specifically?
                    # tqdm.write(f"Row {result['row_id']}: [OK]")

                # CIRCUIT BREAKER CHECK
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    tqdm.write("\n!!! CIRCUIT BREAKER TRIGGERED !!!")
                    tqdm.write(f"Stopped after {MAX_CONSECUTIVE_ERRORS} consecutive timeouts.")
                    tqdm.write("The API seems to be down or overloaded. Please try again later.")
                    break
                
                # Write to file
                writer.writerow(result)
                f.flush()
                pbar.update(1)

    print(f"\nUpdated {OUTPUT_FILE}")

if __name__ == "__main__":
    main()