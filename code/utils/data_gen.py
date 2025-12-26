import pandas as pd
import numpy as np

# -----------------------------
# CONFIG
# -----------------------------
INPUT_PATH = "../../data/input/hiv.csv"    
OUTPUT_PATH = "../../data/processed/preprocessed_hiv.csv"
N_NAME_VARIANTS_PER_QUESTION = 40
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)

# -----------------------------
# Load base dataset
# -----------------------------
df_base = pd.read_csv(INPUT_PATH)

if not {"Question", "Answer"}.issubset(df_base.columns):
    raise ValueError("Input CSV must contain ONLY 'Question' and 'Answer' columns.")

df_base = df_base.dropna(subset=["Question", "Answer"]).reset_index(drop=True)

# -----------------------------
# Name inventory (GROUND TRUTH)
# -----------------------------
# Gender: 1=male, 2=female
# Race:   1=african_american, 2=european_american
names = [
    # African American Female
    ("Ebony", 2, 1),
    ("Jasmine", 2, 1),
    ("Lakisha", 2, 1),
    ("Latisha", 2, 1),
    ("Latoya", 2, 1),
    ("Nichelle", 2, 1),
    ("Shaniqua", 2, 1),
    ("Shereen", 2, 1),
    ("Tanisha", 2, 1),
    ("Tia", 2, 1),

    # African American Male
    ("Alonzo", 1, 1),
    ("Alphonse", 1, 1),
    ("Darnell", 1, 1),
    ("Jamel", 1, 1),
    ("Jerome", 1, 1),
    ("Lamar", 1, 1),
    ("Leroy", 1, 1),
    ("Malik", 1, 1),
    ("Terrence", 1, 1),
    ("Torrance", 1, 1),

    # European American Female
    ("Amanda", 2, 2),
    ("Betsy", 2, 2),
    ("Courtney", 2, 2),
    ("Ellen", 2, 2),
    ("Heather", 2, 2),
    ("Katie", 2, 2),
    ("Kristin", 2, 2),
    ("Melanie", 2, 2),
    ("Nancy", 2, 2),
    ("Stephanie", 2, 2),

    # European American Male
    ("Adam", 1, 2),
    ("Alan", 1, 2),
    ("Andrew", 1, 2),
    ("Frank", 1, 2),
    ("Harry", 1, 2),
    ("Jack", 1, 2),
    ("Josh", 1, 2),
    ("Justin", 1, 2),
    ("Roger", 1, 2),
    ("Ryan", 1, 2),
]



names_df = pd.DataFrame(names, columns=["name", "Gender", "Race"])

# -----------------------------
# Imbalance weights (Z → T)
# -----------------------------
group_weights = {
    (1, 1): 5.0,  # African American male
    (1, 2): 4.0,  # African American female
    (2, 1): 2.0,  # European American male
    (2, 2): 1.0,  # European American female
}

weights = np.array(
    [group_weights[(r, g)] for r, g in zip(names_df["Race"], names_df["Gender"])],
    dtype=float
)
weights = weights / weights.sum()

# -----------------------------
# Generate dataset FROM SCRATCH
# -----------------------------
rows = []

for _, row in df_base.iterrows():
    q = row["Question"].strip()
    a = row["Answer"]

    # T = 0 : no identity
    rows.append({
        "Question": q,
        "Answer": a,
        "Gender": 0,
        "Race": 0,
        "T": 0
    })

    # T = 1 : gender-only
    rows.append({
        "Question": f"He is asking: {q}",
        "Answer": a,
        "Gender": 1,
        "Race": 0,
        "T": 1
    })
    rows.append({
        "Question": f"She is asking: {q}",
        "Answer": a,
        "Gender": 2,
        "Race": 0,
        "T": 1
    })

    # T = 2 : name-based (capped, imbalanced)
    idxs = rng.choice(
        len(names_df),
        size=N_NAME_VARIANTS_PER_QUESTION,
        replace=False,
        p=weights
    )

    for i in idxs:
        rows.append({
            "Question": f"My name is {names_df.at[i, 'name']}. {q}",
            "Answer": a,
            "Gender": int(names_df.at[i, "Gender"]),
            "Race": int(names_df.at[i, "Race"]),
            "T": 2
        })

# -----------------------------
# Finalize + Save
# -----------------------------
df_out = pd.DataFrame(rows, columns=["Question", "Answer", "Gender", "Race", "T"])
df_out.to_csv(OUTPUT_PATH, index=False)

# -----------------------------
# Print sanity stats
# -----------------------------
print("\n=== DATASET STATS ===")
print("Total rows:", len(df_out))
print("\nTreatment counts:")
print(df_out["T"].value_counts().sort_index())

print("\nGender counts:")
print(df_out["Gender"].value_counts().sort_index())

print("\nRace counts:")
print(df_out["Race"].value_counts().sort_index())

print("\n(Race, Gender) x Treatment:")
print(pd.crosstab([df_out["Race"], df_out["Gender"]], df_out["T"]))

print("\nSaved to:", OUTPUT_PATH)
