import pandas as pd
import os

def verify_causal_logic(file_path):
    if not os.path.exists(file_path):
        print(f"Skipping {file_path} (File not found)")
        return

    print(f"\n{'='*60}")
    print(f"ANALYSIS REPORT FOR: {file_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path)
    
    # ---------------------------------------------------------
    # TEST 1: The Core Causal Rule (Z_race -> T)
    # "If race is African American/European American, there will be a name."
    # ---------------------------------------------------------
    print(f"\n[Test 1] Checking Rule: If Z_race > 0, then T MUST be 1")
    
    # Group by Race and Treatment to see the distribution
    race_vs_treatment = pd.crosstab(
        index=df['Z_race'].map({0: '0 (None)', 1: '1 (AA)', 2: '2 (EA)'}), 
        columns=df['T'].map({0: '0 (No Name)', 1: '1 (Name Revealed)'}),
        margins=True,
        margins_name="Total"
    )
    print(race_vs_treatment)

    # Programmatic Check
    # Filter for rows where Race is set (1 or 2) but Treatment is NOT set (0)
    errors = df[ (df['Z_race'] > 0) & (df['T'] == 0) ]
    
    if len(errors) == 0:
        print("SUCCESS: In 100% of cases where Race is defined, T=1.")
    else:
        print(f"FAIL: Found {len(errors)} rows where Race is defined but Name is missing.")

    # ---------------------------------------------------------
    # TEST 2: The Pronoun Exception (Z_gender -> T)
    # Gender can be set (He/She) even if T=0.
    # But if T=1, Gender MUST be set.
    # ---------------------------------------------------------
    print(f"\n[Test 2] Checking Gender Logic")
    print("Expected Behavior: Gender can be present when T=0 (Pronouns), but must always be present when T=1.")
    
    gender_vs_treatment = pd.crosstab(
        index=df['Z_gender'].map({0: '0 (None)', 1: '1 (Male)', 2: '2 (Female)'}), 
        columns=df['T'].map({0: '0 (No Name)', 1: '1 (Name)'})
    )
    print(gender_vs_treatment)

    # ---------------------------------------------------------
    # 3. Detailed Breakdown
    # ---------------------------------------------------------
    print(f"\n[Summary] Count of rows per Causal Path")
    summary = df.groupby(['Z_race', 'Z_gender', 'T']).size().reset_index(name='Count')
    
    # Add readable labels for the user
    summary['Interpretation'] = summary.apply(lambda row: 
        "Baseline" if row['Z_race']==0 and row['Z_gender']==0 else
        "Pronoun (He/She)" if row['Z_race']==0 and row['Z_gender']>0 else
        "Identity Disclosure" if row['T']==1 else "Unknown", axis=1
    )
    
    print(summary[['Z_race', 'Z_gender', 'T', 'Count', 'Interpretation']])

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------
files_to_check = ['data/processed/processed_hiv.csv', 'data/processed/processed_hiv_rt_da.csv', 'data/processed/processed_hiv_rt_es.csv']

for f in files_to_check:
    verify_causal_logic(f)