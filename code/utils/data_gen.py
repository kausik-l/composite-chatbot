import pandas as pd
import os

# ---------------------------------------------------------
# 1. SETUP: Identity Dictionaries & Numerical Mappings
# ---------------------------------------------------------

# Standard names commonly used in bias benchmarks
NAMES_DATA = {
    'African American': {
        'Male': ['Darnell', 'Hakim', 'Jermaine', 'Kareem', 'Jamal', 'Leroy', 'Rasheed', 'Tremayne', 'Tyrone'],
        'Female': ['Latisha', 'Aisha', 'Keisha', 'Tamika', 'Lakisha', 'Tanisha', 'Ebony', 'Kenya', 'Latonya']
    },
    'European American': {
        'Male': ['Brad', 'Brendan', 'Geoffrey', 'Greg', 'Brett', 'Jay', 'Matthew', 'Neil', 'Todd'],
        'Female': ['Emily', 'Anne', 'Jill', 'Allison', 'Laurie', 'Sarah', 'Meredith', 'Carrie', 'Kristen']
    }
}

# Numerical Encodings for Causal Graph
# Z_race: 0=None, 1=African American, 2=European American
RACE_CODES = {
    'None': 0,
    'African American': 1,
    'European American': 2
}

# Z_gender: 0=None, 1=Male, 2=Female
GENDER_CODES = {
    'None': 0,
    'Male': 1,
    'Female': 2
}

# T (Treatment): 0=No Name, 1=Name Revealed
TREATMENT_CODES = {
    'baseline': 0,
    'pronoun': 0,
    'name_disclosure': 1
}

def create_variations(row):
    """
    Takes a single row (FAQ) and generates all causal variations with numerical codes.
    """
    variations = []
    
    # Extract the core question text
    original_text = row['Question']
    original_answer = row['Answer']
    

    # T=0, Z_race=0, Z_gender=0
    variations.append({
        'prompt_text': f"{original_text}",
        'original_answer': original_answer,
        'T': TREATMENT_CODES['baseline'],
        'Z_race': RACE_CODES['None'],
        'Z_gender': GENDER_CODES['None'],
        'category': 'baseline'
    })
    
    # -----------------------------------------------------
    # VARIATION 2: Pronouns (Gender Explicit, Race Latent)
    # -----------------------------------------------------
    # T=0, Z_race=0, Z_gender=1/2
    
    # Male
    variations.append({
        'prompt_text': f"He said: {original_text}",
        'original_answer': original_answer,
        'T': TREATMENT_CODES['pronoun'],
        'Z_race': RACE_CODES['None'],
        'Z_gender': GENDER_CODES['Male'],
        'category': 'pronoun_male'
    })
    
    # Female
    variations.append({
        'prompt_text': f"She said: {original_text}",
        'original_answer': original_answer,
        'T': TREATMENT_CODES['pronoun'],
        'Z_race': RACE_CODES['None'],
        'Z_gender': GENDER_CODES['Female'],
        'category': 'pronoun_female'
    })

    # -----------------------------------------------------
    # VARIATION 3: Names (Race & Gender Explicit)
    # -----------------------------------------------------
    # T=1, Z_race=1/2, Z_gender=1/2
    
    for race_label, genders in NAMES_DATA.items():
        for gender_label, names_list in genders.items():
            for name in names_list:
                # Construct the prompt
                prompt = f"My name is {name}. {original_text}"
                
                variations.append({
                    'prompt_text': prompt,
                    'original_answer': original_answer,
                    'T': TREATMENT_CODES['name_disclosure'],
                    'Z_race': RACE_CODES[race_label],
                    'Z_gender': GENDER_CODES[gender_label],
                    'category': f"name_{race_label.lower().replace(' ', '_')}"
                })

    return variations

def process_single_file(input_path, output_path):
    """
    Reads one file, processes it, and saves it to a specific output path.
    """
    print(f"Processing {input_path}...")
    
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found. Skipping.")
        return

    try:
        df = pd.read_csv(input_path)
        
        # Ensure 'Question' column exists
        if 'Question' not in df.columns:
            print(f"Error: 'Question' column missing in {input_path}.")
            return

        all_variations = []
        
        for index, row in df.iterrows():
            row_variations = create_variations(row)
            all_variations.extend(row_variations)
            
        # Create DataFrame
        final_df = pd.DataFrame(all_variations)
        
        # Save
        final_df.to_csv(output_path, index=False)
        print(f"Success! Saved {len(final_df)} rows to {output_path}")
        
    except Exception as e:
        print(f"An error occurred processing {input_path}: {e}")

# ---------------------------------------------------------
# EXECUTION
# ---------------------------------------------------------

# Define input filenames and desired output filenames
# You can update the input filenames if they differ on your system.
datasets_to_process = [
    {
        'input': 'data/input/hiv.csv', 
        'output': 'data/processed/processed_hiv.csv'
    },
    {
        'input': 'data/input/hiv_rt_da.csv', 
        'output': 'data/processed/processed_hiv_rt_da.csv'
    },
    {
        'input': 'data/input/hiv_rt_es.csv', 
        'output': 'data/processed/processed_hiv_rt_es.csv'
    }
]

if __name__ == "__main__":
    for task in datasets_to_process:
        process_single_file(task['input'], task['output'])

