# 🃏 Model Card — Personalized News Recommendation with Contextual Bandits

> **Model Card Version:** 1.0  
> **Date:** 2025  
> **Institution:** Holy Angel University, Angeles City, Pampanga  
> **Authors:** Jero E. Halili, Kian Andrei G. James, Gerail C. Mendoza, Angelica Z. Tinio

---

## 📌 Model Overview

This project is a **hybrid recommendation system** that combines Natural Language Processing (NLP) text embeddings with Reinforcement Learning (RL) contextual bandit algorithms to deliver personalized news article recommendations.

It does **not** serve as a single monolithic model — instead, it is a **pipeline of cooperating components**:

| Component | Purpose |
|---|---|
| TF-IDF + SVD | Sparse keyword-based article representation |
| Sentence Transformer (SBERT) | Dense semantic article representation |
| 1D-CNN (SBERT input) | Fast feature extraction from SBERT embeddings |
| 1D-CNN from scratch | Ablation baseline without pretrained embeddings |
| ε-Greedy Bandit | Adaptive article selection via exploration/exploitation |
| LinUCB Bandit | Upper confidence bound-based adaptive selection |

---

## 🎯 Intended Use

### ✅ Intended Uses
- Academic research on hybrid NLP + RL recommendation systems
- Offline benchmarking of contextual bandit algorithms for content recommendation
- Educational demonstration of exploration–exploitation tradeoffs in news ranking

### ❌ Out-of-Scope Uses
- Production deployment without real user interaction data and privacy safeguards
- Medical, legal, financial, or safety-critical recommendation tasks
- Real-time personalization without retraining on live user feedback
- Any use case involving real, identified user behavioral data without consent

---

## 📦 Training Data

| Property | Details |
|---|---|
| **Dataset** | News Category Dataset (Misra, 2022) |
| **Source** | Kaggle (publicly available) |
| **Size** | ~200,000 HuffPost news articles |
| **Date Range** | 2012–2022 |
| **Categories** | 42 (e.g., Politics, Entertainment, Wellness, Travel, World News) |
| **Fields Used** | Headline, short description, category, author, publication date |
| **User Data** | None — dataset contains only published article metadata |
| **Privacy Risk** | Minimal — no user-identifying information present |

### ⚠️ Known Data Limitations
- Sourced exclusively from HuffPost, which may reflect editorial bias toward certain political and cultural perspectives
- Some categories are more represented than others (class imbalance), mitigated via oversampling in the pipeline
- Articles from 2012–2022 may not reflect current writing styles, topics, or news formats
- Simulated user interactions — no real click-through behavior used

---

## 🏗️ Model Architecture

```
Article Text (Headline + Description)
        ↓
  Preprocessing (tokenization, cleaning)
        ↓
  Embedding Layer (choose one):
  ├── TF-IDF + SVD  → sparse vector (keyword-based)
  └── SBERT         → 384-dim dense vector (semantic)
        ↓
  Optional CNN Feature Extractor
  ├── Embedding → Conv1D (128 filters, kernel=5)
  ├── Global Max Pooling
  └── Dense layer
        ↓
  Cosine Similarity Ranking (initial ranking)
        ↓
  Contextual Bandit Agent (choose one):
  ├── ε-Greedy (ε = 0.1)
  └── LinUCB (upper confidence bound)
        ↓
  Reward Signal: +1 (simulated click) / 0 (no click)
        ↓
  Updated Policy → Next Recommendation
```

---

## 📊 Performance

### NLP Classification (Category Prediction)

| Model | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| SBERT + Logistic Regression | **88.51%** | **86.89%** | Best overall classifier |
| EmbeddingCNN (SBERT input) | 83.91% | 81.96% | Fast convergence, best val_acc = 89.66% |
| TextCNN from Scratch | 64.37% | 58.33% | Ablation baseline only |

### Bandit Agent Performance (Ablation: Embedding × Algorithm)

| Embedding | Algorithm | Mean Reward/Step |
|---|---|---|
| TF-IDF + SVD | LinUCB | **0.1840** ✅ |
| TF-IDF + SVD | ε-Greedy | 0.1720 |
| SBERT | LinUCB | 0.0640 |
| SBERT | ε-Greedy | 0.0560 |
| Random (baseline) | — | 0.0517 |

### Cumulative Reward (3,000 Steps)

| Agent | Cumulative Reward | Rolling Mean |
|---|---|---|
| ε-Greedy | ~180 | ~0.080 |
| LinUCB | ~190 | ~0.063 |
| Random | ~154 | ~0.054 |

### Evaluation Metrics Used

| Metric | Type | Description |
|---|---|---|
| nDCG | Ranking | Penalizes relevant items ranked too low |
| Hit@K | Ranking | Whether a relevant article appears in top K |
| Cumulative Reward | RL | Total simulated clicks across all steps |
| Rolling Avg Reward | RL | Smoothed per-step reward trend |
| Accuracy / Macro-F1 | Classification | Category prediction quality |

---

## ⚖️ Factors and Disaggregated Evaluation

### Instrumented Categories (Confusion Matrix)
The system was explicitly evaluated across the following categories:
- **Entertainment** — High accuracy; well-represented in training data
- **Politics** — Good accuracy; slight confusion with World News
- **World News** — Moderate accuracy; occasional confusion with Politics and U.S. News
- **Crime, Culture & Arts, Environment, Sports, U.S. News** — Evaluated in multi-class runs

### Known Failure Modes
- Semantically adjacent categories (e.g., Politics ↔ World News) are more likely to be misclassified
- Rare categories with fewer training samples tend to have lower recall
- CNN from scratch struggles with long-tail and context-dependent articles

---

## 🔁 Caveats & Recommendations

- **Do not use TF-IDF + LinUCB results alone to claim superiority** — SBERT's advantages emerge in real semantic retrieval tasks; LinUCB's linear structure simply fits TF-IDF vectors more naturally in offline simulation
- **Offline simulation ≠ real-world performance** — Simulated clicks are a proxy; real CTR data would change reward distributions significantly
- **Retrain periodically** — News topics drift over time; models trained on 2012–2022 data will degrade on future content without retraining
- **Add diversity constraints** before any production deployment to prevent filter bubbles

---

## 📋 Model Details

| Property | Value |
|---|---|
| **Framework** | Python, scikit-learn, sentence-transformers, PyTorch |
| **SBERT Model** | `all-MiniLM-L6-v2` (384-dim output) |
| **CNN Filters** | 128, kernel size 5 |
| **Bandit ε** | 0.1 (10% exploration) |
| **Simulation Steps** | 3,000 per run, 6 total runs |
| **Random Seeds** | Multiple seeds averaged for reliability |
| **License** | MIT |

---

## 📚 Citation

If you use this work, please cite:

```
Halili, J. E., James, K. A. G., Mendoza, G. C., & Tinio, A. Z. (2025).
Personalized News Recommendation with Contextual Bandits Using NLP Embedding.
Holy Angel University, Angeles City, Pampanga.
```
