# notebooks/01_eda.py
# Run with: jupyter nbconvert --to notebook --execute 01_eda.py
# Or simply run as a Python script: python notebooks/01_eda.py
"""
Exploratory Data Analysis — News Category Dataset
Covers:
  - Category distribution
  - Text length distribution
  - Word clouds per top category
  - Train/val/test split balance check
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from collections import Counter

DATA_DIR    = Path("data/processed")
RAW_FILE    = Path("data/News_Category_Dataset_v3.json")
RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Load processed splits ──────────────────────────────────────────────────
train = pd.read_csv(DATA_DIR / "train.csv").fillna("")
val   = pd.read_csv(DATA_DIR / "val.csv").fillna("")
test  = pd.read_csv(DATA_DIR / "test.csv").fillna("")

print(f"Train: {len(train):,}  Val: {len(val):,}  Test: {len(test):,}")
print(f"Categories: {train['category'].nunique()}")

# ── 2. Category distribution ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

cat_counts = train["category"].value_counts()
colors = plt.cm.tab20(np.linspace(0, 1, len(cat_counts)))

axes[0].barh(cat_counts.index[::-1], cat_counts.values[::-1], color=colors[::-1])
axes[0].set_title("Category Distribution (Train)", fontsize=13)
axes[0].set_xlabel("Article count")
axes[0].tick_params(axis="y", labelsize=9)

# Class balance across splits
split_balance = pd.DataFrame({
    "Train": train["category"].value_counts(normalize=True),
    "Val":   val["category"].value_counts(normalize=True),
    "Test":  test["category"].value_counts(normalize=True),
}).fillna(0)
split_balance.plot(kind="bar", ax=axes[1], width=0.7)
axes[1].set_title("Split Balance (normalised proportions)", fontsize=13)
axes[1].set_xlabel("")
axes[1].tick_params(axis="x", rotation=45, labelsize=8)
axes[1].legend(fontsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_category_distribution.png", dpi=120)
plt.close()
print("Saved: eda_category_distribution.png")

# ── 3. Text length distribution ───────────────────────────────────────────────
train["text_len"] = train["clean_text"].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))

axes[0].hist(train["text_len"], bins=60, color="steelblue", edgecolor="white")
axes[0].axvline(train["text_len"].median(), color="red",
                linestyle="--", label=f"Median={train['text_len'].median():.0f}")
axes[0].set_title("Text Length Distribution (words)")
axes[0].set_xlabel("Number of words")
axes[0].legend()

# Per-category median length
med_len = train.groupby("category")["text_len"].median().sort_values()
axes[1].barh(med_len.index, med_len.values, color="darkorange")
axes[1].set_title("Median Text Length per Category")
axes[1].set_xlabel("Median word count")
axes[1].tick_params(axis="y", labelsize=9)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_text_length.png", dpi=120)
plt.close()
print("Saved: eda_text_length.png")

# ── 4. Top words per category ─────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings("ignore")

top_cats  = train["category"].value_counts().head(6).index.tolist()
fig, axes = plt.subplots(2, 3, figsize=(16, 8))
axes      = axes.flatten()

for i, cat in enumerate(top_cats):
    cat_texts = train[train["category"] == cat]["clean_text"].tolist()
    tfidf = TfidfVectorizer(max_features=15, stop_words="english")
    try:
        tfidf.fit(cat_texts)
        words  = list(tfidf.vocabulary_.keys())
        scores = tfidf.idf_
        sorted_words = sorted(zip(words, [tfidf.idf_[tfidf.vocabulary_[w]]
                                          for w in words]),
                              key=lambda x: x[1])
        top_w, top_s = zip(*sorted_words[:12])
        axes[i].barh(list(top_w)[::-1], [1/s for s in top_s[::-1]],
                     color=plt.cm.Paired(i/len(top_cats)))
        axes[i].set_title(f"Top words: {cat}", fontsize=10)
        axes[i].tick_params(axis="y", labelsize=8)
    except Exception:
        axes[i].set_title(f"{cat} (insufficient data)")

plt.suptitle("Top TF-IDF Words per Category (train set)", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_top_words.png", dpi=120, bbox_inches="tight")
plt.close()
print("Saved: eda_top_words.png")

# ── 5. Summary statistics ─────────────────────────────────────────────────────
summary = {
    "total_articles":    int(len(train) + len(val) + len(test)),
    "n_categories":      int(train["category"].nunique()),
    "train_size":        int(len(train)),
    "val_size":          int(len(val)),
    "test_size":         int(len(test)),
    "avg_text_len_words": float(train["text_len"].mean()),
    "median_text_len":   float(train["text_len"].median()),
    "min_cat_articles":  int(train["category"].value_counts().min()),
    "max_cat_articles":  int(train["category"].value_counts().max()),
}
with open(RESULTS_DIR / "eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)
print("\nEDA Summary:")
for k, v in summary.items():
    print(f"  {k:<28}: {v}")
print("\nEDA complete. All plots saved to experiments/results/")
