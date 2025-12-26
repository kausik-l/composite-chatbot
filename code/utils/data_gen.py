import pandas as pd
import numpy as np


INPUT_PATH = "../../data/input/hiv.csv"
OUTPUT_PATH = "../../data/processed/processed_hiv.csv"
N_NAME_VARIANTS_PER_QUESTION = 6
RANDOM_SEED = 42

rng = np.random.default_rng(RANDOM_SEED)


df_base = pd.read_csv(INPUT_PATH)
df_base = df_base.dropna(subset=["Question", "Answer"]).reset_index(drop=True)


# Gender: 1=male, 2=female
# Race:   1=african_american, 2=european_american
names = [
    # African American Female
    ("Ebony", 2, 1), ("Jasmine", 2, 1), ("Lakisha", 2, 1),
    ("Latisha", 2, 1), ("Latoya", 2, 1),

    # African American Male
    ("Alonzo", 1, 1), ("Darnell", 1, 1), ("Jamel", 1, 1),
    ("Leroy", 1, 1), ("Malik", 1, 1),

    # European American Female
    ("Amanda", 2, 2), ("Heather", 2, 2), ("Katie", 2, 2),

    # European American Male
    ("Adam", 1, 2), ("Andrew", 1, 2), ("Ryan", 1, 2),
]

names_df = pd.DataFrame(names, columns=["name", "Gender", "Race"])


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

    # T = 1 : gender only
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

    # T = 2 : gender + race (capped, imbalanced)
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


df = pd.DataFrame(rows, columns=["Question", "Answer", "Gender", "Race", "T"])
df.to_csv(OUTPUT_PATH, index=False)


print("\n================ DATASET SUMMARY ================\n")
print("Total rows:", len(df))
print("Total base FAQs:", len(df_base))

print("\n--- Treatment distribution (T) ---")
print(df["T"].value_counts().sort_index())

print("\n--- Gender distribution ---")
print(df["Gender"].value_counts().sort_index())

print("\n--- Race distribution ---")
print(df["Race"].value_counts().sort_index())

print("\n--- (Race, Gender) intersection counts ---")
print(df.groupby(["Race", "Gender"]).size().sort_values(ascending=False))

print("\n--- Treatment by Gender (row-normalized) ---")
print(pd.crosstab(df["Gender"], df["T"], normalize="index"))

print("\n--- Treatment by Race (row-normalized) ---")
print(pd.crosstab(df["Race"], df["T"], normalize="index"))

print("\n--- Treatment by (Race, Gender) ---")
print(pd.crosstab([df["Race"], df["Gender"]], df["T"]))

print("\nSaved dataset to:", OUTPUT_PATH)
print("\n=================================================\n")
