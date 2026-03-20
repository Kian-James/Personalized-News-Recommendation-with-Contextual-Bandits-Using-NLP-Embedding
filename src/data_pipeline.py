"""
data_pipeline.py
Load, clean, and split the HuffPost News Category Dataset.
Dataset: https://www.kaggle.com/datasets/rmisra/news-category-dataset
Format:  JSON-lines, one article per line.
"""

import json
import random
import re
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── reproducibility ──────────────────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ── constants ─────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
RAW_FILE   = DATA_DIR / "News_Category_Dataset_v3.json"
PROCESSED  = DATA_DIR / "processed"

# Keep top-N categories for a tractable classification problem
TOP_N_CATS = 20
MIN_ARTICLES_PER_CAT = 200


# ── helpers ───────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Lower-case, remove HTML entities & excess whitespace."""
    text = text.lower()
    text = re.sub(r"&[a-z]+;", " ", text)          # HTML entities
    text = re.sub(r"http\S+", " ", text)            # URLs
    text = re.sub(r"[^a-z0-9\s\'\-]", " ", text)   # keep apostrophes/hyphens
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_raw(path: Path = RAW_FILE) -> pd.DataFrame:
    """Read JSON-lines file into a DataFrame."""
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
    """
    1. Combine headline + short_description into a single text field.
    2. Drop rows with empty text or category.
    3. Keep only top-N categories with enough samples.
    4. Clean text.
    """
    df = df.copy()

    # Combine text fields
    df["headline"]          = df["headline"].fillna("").astype(str)
    df["short_description"] = df["short_description"].fillna("").astype(str)
    df["text"] = (df["headline"] + " " + df["short_description"]).str.strip()

    # Drop empties
    df = df[df["text"].str.len() > 5].reset_index(drop=True)

    # Filter to top-N categories
    cat_counts = df["category"].value_counts()
    top_cats   = cat_counts[cat_counts >= MIN_ARTICLES_PER_CAT].head(TOP_N_CATS).index.tolist()
    df = df[df["category"].isin(top_cats)].reset_index(drop=True)

    # Clean text
    df["clean_text"] = df["text"].apply(clean_text)

    # Encode labels
    le = LabelEncoder()
    df["label"] = le.fit_transform(df["category"])

    print(f"[data_pipeline] After filtering: {len(df):,} articles, "
          f"{df['category'].nunique()} categories.")
    print(f"  Categories: {sorted(df['category'].unique())}")
    return df, le


def split_data(df: pd.DataFrame, val_size=0.10, test_size=0.10):
    """
    Stratified split: train / val / test.
    No leakage — splits are done once and indices saved.
    """
    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=SEED
    )
    relative_val = val_size / (1 - test_size)
    train, val = train_test_split(
        train_val, test_size=relative_val, stratify=train_val["label"], random_state=SEED
    )
    print(f"[data_pipeline] Split → train={len(train):,} | "
          f"val={len(val):,} | test={len(test):,}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def run_pipeline(raw_path: Path = RAW_FILE):
    """End-to-end: load → preprocess → split → save."""
    PROCESSED.mkdir(parents=True, exist_ok=True)

    df_raw        = load_raw(raw_path)
    df, label_enc = preprocess(df_raw)
    train, val, test = split_data(df)

    train.to_csv(PROCESSED / "train.csv", index=False)
    val.to_csv(PROCESSED  / "val.csv",   index=False)
    test.to_csv(PROCESSED / "test.csv",  index=False)

    # Save label mapping
    mapping = dict(enumerate(label_enc.classes_))
    with open(PROCESSED / "label_map.json", "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"[data_pipeline] Saved splits to {PROCESSED}/")
    return train, val, test, label_enc


if __name__ == "__main__":
    run_pipeline()
