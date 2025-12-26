import pandas as pd
import re
from transformers import MarianMTModel, MarianTokenizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Load processed dataset
# -----------------------------
input_path = "../../data/processed/hiv.csv"
df = pd.read_csv(input_path)

assert set(df.columns) == {"Question", "Answer", "Gender", "Race"}


def load_mt(model_name):
    tok = MarianTokenizer.from_pretrained(model_name)
    mdl = MarianMTModel.from_pretrained(model_name).to(device)
    return tok, mdl

# Spanish
tok_en_es, mdl_en_es = load_mt("Helsinki-NLP/opus-mt-en-es")
tok_es_en, mdl_es_en = load_mt("Helsinki-NLP/opus-mt-es-en")

# Danish
tok_en_da, mdl_en_da = load_mt("Helsinki-NLP/opus-mt-en-da")
tok_da_en, mdl_da_en = load_mt("Helsinki-NLP/opus-mt-da-en")


def translate(texts, tokenizer, model):
    batch = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        gen = model.generate(**batch, max_length=256)
    return tokenizer.batch_decode(gen, skip_special_tokens=True)

def split_prefix(question):
    """
    Extract identity prefix if present.
    """
    patterns = [
        r"^(My name is [A-Za-z]+\. )",
        r"^(He is asking: )",
        r"^(She is asking: )"
    ]
    for p in patterns:
        m = re.match(p, question)
        if m:
            return m.group(1), question[m.end():]
    return "", question


def backtranslate_es(texts):
    es = translate(texts, tok_en_es, mdl_en_es)
    en = translate(es, tok_es_en, mdl_es_en)
    return en

def backtranslate_da(texts):
    da = translate(texts, tok_en_da, mdl_en_da)
    en = translate(da, tok_da_en, mdl_da_en)
    return en


def paraphrase_dataset(df, backtranslate_fn):
    prefixes = []
    cores = []

    for q in df["Question"]:
        p, c = split_prefix(q)
        prefixes.append(p)
        cores.append(c)

    paraphrased_cores = backtranslate_fn(cores)

    new_questions = [
        p + c for p, c in zip(prefixes, paraphrased_cores)
    ]

    df_new = df.copy()
    df_new["Question"] = new_questions
    return df_new


df_es = paraphrase_dataset(df, backtranslate_es)
df_da = paraphrase_dataset(df, backtranslate_da)


es_path = "../../data/processed/hiv_es.csv"
da_path = "../../data/processed/hiv_da.csv"

df_es.to_csv(es_path, index=False)
df_da.to_csv(da_path, index=False)

print("Saved:")
print(es_path)
print(da_path)
