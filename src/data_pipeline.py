"""
data_pipeline.py
Load, clean, and split the HuffPost News Category Dataset.
Dataset: https://www.kaggle.com/datasets/rmisra/news-category-dataset

Key design decisions for 1,000-article subset:
  - Merge semantically overlapping categories to reduce confusion
  - Raise MIN_ARTICLES_PER_CAT so every class has enough samples
  - Oversample minority classes so the CNN trains on balanced data
"""

import json
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

DATA_DIR  = Path("data")
RAW_FILE  = DATA_DIR / "News_Category_Dataset_v3.json"
PROCESSED = DATA_DIR / "processed"

# ── Raise minimum so every class has enough signal ────────────────────────────
TOP_N_CATS           = 6     # fewer, cleaner categories
MIN_ARTICLES_PER_CAT = 50    # guarantee enough per class

# ── Merge categories that are semantically too similar ────────────────────────
# U.S. NEWS and POLITICS overlap heavily → merge into POLITICS
# CRIME has too few samples → merge into U.S. NEWS (then both go to POLITICS)
# CULTURE & ARTS and ENVIRONMENT have too few → drop (below MIN threshold)
CATEGORY_MAP = {
    "U.S. NEWS":      "POLITICS",       # overlap too high
    "CRIME":          "POLITICS",       # domestic political/legal
    "CULTURE & ARTS": "ENTERTAINMENT",  # broad entertainment umbrella
    "ARTS":           "ENTERTAINMENT",
    "ARTS & CULTURE": "ENTERTAINMENT",
    "COMEDY":         "ENTERTAINMENT",
    "WEIRD NEWS":     "ENTERTAINMENT",
    "STYLE":          "ENTERTAINMENT",
    "STYLE & BEAUTY": "ENTERTAINMENT",
    "TASTE":          "ENTERTAINMENT",
    "FOOD & DRINK":   "ENTERTAINMENT",
    "HOME & LIVING":  "ENTERTAINMENT",
    "PARENTS":        "ENTERTAINMENT",
    "WEDDINGS":       "ENTERTAINMENT",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"&[a-z]+;", " ", text)
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s\'\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    df = pd.DataFrame(records)
    print(f"[data_pipeline] Loaded {len(df):,} raw articles, "
          f"{df['category'].nunique()} categories.")
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Combine text
    df["headline"]          = df["headline"].fillna("").astype(str)
    df["short_description"] = df["short_description"].fillna("").astype(str)
    df["text"] = (df["headline"] + " " + df["short_description"]).str.strip()
    df = df[df["text"].str.len() > 5].reset_index(drop=True)

    # Apply category merges
    df["category"] = df["category"].replace(CATEGORY_MAP)

    # Keep top-N categories with enough samples
    cat_counts = df["category"].value_counts()
    top_cats   = cat_counts[cat_counts >= MIN_ARTICLES_PER_CAT]\
                     .head(TOP_N_CATS).index.tolist()
    df = df[df["category"].isin(top_cats)].reset_index(drop=True)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])

    print(f"[data_pipeline] After filtering: {len(df):,} articles, "
          f"{df['category'].nunique()} categories.")
    print(f"  Categories: {sorted(df['category'].unique())}")
    print(f"  Class distribution:\n"
          + "\n".join(f"    {cat}: {n}"
                      for cat, n in df["category"].value_counts().items()))
    return df, le


def oversample_minority(df: pd.DataFrame, random_state: int = SEED) -> pd.DataFrame:
    """
    Oversample minority classes to the count of the majority class.
    Only applied to the training split — never val or test.
    """
    max_count = df["category"].value_counts().max()
    parts = []
    for cat, group in df.groupby("category"):
        if len(group) < max_count:
            extra = group.sample(n=max_count - len(group),
                                 replace=True, random_state=random_state)
            parts.append(pd.concat([group, extra]))
        else:
            parts.append(group)
    balanced = pd.concat(parts).sample(frac=1, random_state=random_state)\
                               .reset_index(drop=True)
    print(f"[data_pipeline] After oversampling: {len(balanced):,} train samples "
          f"({balanced['category'].value_counts().to_dict()})")
    return balanced


def split_data(df, val_size=0.10, test_size=0.10):
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=SEED)
    rel_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=rel_val, stratify=train_val["label"], random_state=SEED)
    print(f"[data_pipeline] Split → train={len(train):,} | "
          f"val={len(val):,} | test={len(test):,}")
    return (train.reset_index(drop=True),
            val.reset_index(drop=True),
            test.reset_index(drop=True))


def run_pipeline(raw_path: Path = RAW_FILE):
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df_raw        = load_raw(raw_path)
    df, label_enc = preprocess(df_raw)
    train, val, test = split_data(df)

    # Oversample training split only
    train_balanced = oversample_minority(train)

    train_balanced.to_csv(PROCESSED / "train.csv", index=False)
    val.to_csv(PROCESSED            / "val.csv",   index=False)
    test.to_csv(PROCESSED           / "test.csv",  index=False)

    mapping = dict(enumerate(label_enc.classes_))
    with open(PROCESSED / "label_map.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[data_pipeline] Saved splits to {PROCESSED}/")
    return train_balanced, val, test, label_enc


if __name__ == "__main__":
    run_pipeline()