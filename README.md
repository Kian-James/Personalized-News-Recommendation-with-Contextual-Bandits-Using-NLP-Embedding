GitHub Release v0.1

# 📰 Personalized News Recommendation with Contextual Bandits Using NLP Embedding

[![Python Version](https://img.shields.io/badge/python->=2.7.13-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

> A hybrid NLP + Reinforcement Learning system that learns to recommend personalized news articles using contextual multi-armed bandits and semantic text embeddings.

---

## 🧠 Overview

This project tackles the **information overload problem** on news platforms. Instead of static ranking, the system continuously learns user preferences through simulated interactions — getting smarter with every click.

**Core idea:** Represent articles as semantic vectors → rank them → let a bandit agent learn which ones users actually click → repeat.

---

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/your-repo/news-recommendation-bandits.git
cd news-recommendation-bandits
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the Dataset

Get the [News Category Dataset](https://www.kaggle.com/datasets/rmisra/news-category-dataset) from Kaggle and place it in the `data/` folder.

```
data/
└── News_Category_Dataset_v3.json
```

### 4. Run the Data Pipeline

```bash
python src/data_pipeline.py
```

Filters and splits ~200,000 HuffPost articles (2012–2022) into train/val/test sets across 42 categories.

### 5. Generate NLP Embeddings

```bash
# TF-IDF + SVD
python src/embed_tfidf.py

# Sentence Transformer (SBERT)
python src/embed_sbert.py
```

Embeddings are saved to `experiments/results/embeddings/`.

### 6. Train the CNN Model (Optional)

```bash
# Using SBERT input (Primary CNN)
python src/train_cnn_sbert.py

# From scratch (Ablation)
python src/train_cnn_scratch.py
```

### 7. Run the Reinforcement Learning Simulation

```bash
python src/run_bandit.py
```

Runs epsilon-greedy and LinUCB agents over 3,000 steps. Results and plots are saved to `experiments/results/`.

### 8. View Ablation Results

```bash
python src/ablation.py
```

Compares all combinations of embedding type × bandit algorithm.

---

## ⚙️ Algorithms

| Algorithm | Description |
|---|---|
| **ε-Greedy** | Exploits best known article `1-ε` of the time, explores randomly `ε` of the time. Default ε = 0.1 |
| **LinUCB** | Selects articles using upper confidence bounds on context vectors. Balances exploration mathematically. |
| **Hybrid LinUCB** | Extends LinUCB by modeling user–article interaction features for richer context. |
| **Random** | Selects articles randomly. Used as the baseline. No learning. |

---

## 📊 Results Summary

### 🔢 NLP Classification Performance

| Model | Accuracy | Macro-F1 |
|---|---|---|
| SBERT + Logistic Regression | **88.51%** | **86.89%** |
| EmbeddingCNN (SBERT input) | 83.91% | 81.96% |
| TextCNN from Scratch | 64.37% | 58.33% |

> **Key insight:** SBERT captures deep semantic meaning. CNN learns fast but misses contextual nuance. TF-IDF only matches keywords, not meaning.

---

### 🎯 Bandit Agent Performance (Mean Reward per Step — Ablation)

| Embedding | Algorithm | Mean Reward/Step |
|---|---|---|
| TF-IDF + SVD | **LinUCB** | **0.1840** ✅ Best |
| TF-IDF + SVD | ε-Greedy | 0.1720 |
| SBERT | LinUCB | 0.0640 |
| SBERT | ε-Greedy | 0.0560 |
| Random (baseline) | — | 0.0517 |

> **Key insight:** TF-IDF + SVD with LinUCB achieved the highest per-step reward. Despite SBERT's superior semantic understanding, TF-IDF's sparse representations were more compatible with LinUCB's linear model assumptions in this offline simulation setup.

---

### 📈 Cumulative Reward Over 3,000 Steps

| Agent | Cumulative Reward | Rolling Avg Reward |
|---|---|---|
| ε-Greedy | ~180 | Mean ≈ 0.080 |
| LinUCB | ~190 | Mean ≈ 0.063 |
| Random | ~154 | Mean ≈ 0.054 |

> Both learned agents **significantly outperform random selection**, confirming that the system improves over time as it receives simulated user feedback.

---

### 🔍 Classification Confusion Summary

The confusion matrix shows strong diagonal performance (correct predictions) for major categories: **Entertainment**, **Politics**, and **World News**. Minor cross-category errors remain, particularly between semantically similar categories — a known challenge in multi-class text classification.

---

## 🏗️ System Architecture

```
Raw Articles (HuffPost Dataset)
        ↓
  Data Pipeline (filter, split, oversample)
        ↓
  NLP Embedding Layer
  ├── TF-IDF + SVD  →  sparse keyword vectors
  └── SBERT         →  dense semantic vectors
        ↓
  Initial Ranking (Cosine Similarity)
        ↓
  Contextual Bandit Agent
  ├── ε-Greedy (explore/exploit with fixed ε)
  └── LinUCB   (explore/exploit with confidence bounds)
        ↓
  Simulated User Feedback (click = +1, no click = 0)
        ↓
  Personalized Recommendations (improving over time)
```

---

## 📐 Evaluation Metrics

| Metric | Type | What it measures |
|---|---|---|
| **nDCG** | Ranking | Quality of article ordering (penalizes bad ranks) |
| **Hit@K** | Ranking | Whether a relevant article appears in top-K results |
| **Cumulative Reward** | RL | Total clicks accumulated over all steps |
| **Rolling Avg Reward** | RL | Recent performance trend of the agent |
| **Accuracy / Macro-F1** | Classification | Category prediction quality |

---

## ⚠️ Ethics & Limitations

- **Filter Bubble Risk:** Click-optimized agents can narrow content diversity. Diversity constraints across the 42 categories are advised.
- **Dataset Bias:** Articles span 2012–2022 HuffPost only; some categories are overrepresented.
- **Reward Bias:** Click = +1 may favor sensational headlines over substantive content.
- **Offline Only:** All evaluation uses simulated interactions — real user data would require anonymized logs and data protection compliance.
- **Opacity:** Sentence Transformer + bandit pipeline lacks explainability; future versions should add recommendation rationale.

---

## 👥 Team

| Name | Role |
|---|---|
| **Angelica Z. Tinio** | Project Lead / System Integration |
| **Gerail C. Mendoza** | Data & Ethics Lead |
| **Kian Andrei G. James** | Modeling Lead (NLP + CNN) |
| **Jero E. Halili** | Evaluation & MLOps Lead |

*Holy Angel University — Angeles City, Pampanga*

---
