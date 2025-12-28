import os
import pandas as pd
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm

# -----------------------------
# ENV FIX (important)
# -----------------------------
os.environ["HF_HUB_DISABLE_XET"] = "1"
os.environ["HF_HOME"] = os.path.expanduser("~/hf_cache")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# PATHS
# -----------------------------
INPUT_PATH = "../../data/input/hiv.csv"  
OUT_ES = "../../data/input/hiv_rt_es_base.csv"
OUT_DA = "../../data/input/hiv_rt_da_base.csv"

# -----------------------------
# Load base FAQs
# -----------------------------
df = pd.read_csv(INPUT_PATH)
df = df.dropna(subset=["Question", "Answer"]).reset_index(drop=True)

questions = df["Question"].tolist()

# -----------------------------
# Translation helpers
# -----------------------------
def load_mt(model_name):
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name).to(DEVICE)
    return tok, mdl


def translate(texts, tok, mdl, batch_size=8, desc="Translating"):
    outputs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=desc):
        batch_texts = texts[i:i+batch_size]
        batch = tok(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            gen = mdl.generate(**batch, max_length=256)

        decoded = tok.batch_decode(gen, skip_special_tokens=True)
        outputs.extend(decoded)

    return outputs


def backtranslate(texts, fwd, back):
    inter = translate(texts, *fwd)
    return translate(inter, *back)

# -----------------------------
# Load MT models
# -----------------------------
en_es = load_mt("Helsinki-NLP/opus-mt-en-es")
es_en = load_mt("Helsinki-NLP/opus-mt-es-en")

en_da = load_mt("Helsinki-NLP/opus-mt-en-da")
da_en = load_mt("Helsinki-NLP/opus-mt-da-en")

# -----------------------------
# Run back-translation
# -----------------------------
df_es = df.copy()
df_es["Question"] = backtranslate(questions, en_es, es_en)

df_da = df.copy()
df_da["Question"] = backtranslate(questions, en_da, da_en)

# -----------------------------
# Save intermediate results
# -----------------------------
os.makedirs(os.path.dirname(OUT_ES), exist_ok=True)

df_es.to_csv(OUT_ES, index=False)
df_da.to_csv(OUT_DA, index=False)

print("Saved base back-translated FAQs:")
print(OUT_ES)
print(OUT_DA)
