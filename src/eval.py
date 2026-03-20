"""
eval.py
Comprehensive evaluation for the News Recommender System.

Covers:
  • Ranking metrics  : nDCG@K, Hit@K
  • Classification   : Accuracy, Macro-F1, per-class confusion matrix
  • RL metrics       : Mean reward, cumulative reward, vs random baseline
  • Ablation studies : TF-IDF vs SBERT embeddings × ε-greedy vs LinUCB
  • Slice analysis   : per-category performance breakdown
  • Error analysis   : top failure cases with text samples

Usage:
    python src/eval.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix,
    classification_report
)

RESULTS_DIR = Path("experiments/results")
DATA_DIR    = Path("data/processed")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Ranking metrics
# ─────────────────────────────────────────────────────────────────────────────

def dcg_at_k(relevances: np.ndarray, k: int) -> float:
    """Discounted Cumulative Gain @k."""
    rels = np.asarray(relevances[:k], dtype=float)
    if len(rels) == 0:
        return 0.0
    discounts = np.log2(np.arange(2, len(rels) + 2))
    return float((rels / discounts).sum())


def ndcg_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Normalised DCG@k."""
    rel = [1.0 if i in relevant else 0.0 for i in recommended[:k]]
    ideal = sorted(rel, reverse=True)
    dcg   = dcg_at_k(rel, k)
    idcg  = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(recommended: list[int], relevant: set[int], k: int) -> float:
    """Hit@k: 1 if any of top-k is relevant."""
    return float(any(i in relevant for i in recommended[:k]))


def evaluate_ranker(item_embeddings: np.ndarray,
                    user_preferences: np.ndarray,
                    k_values: tuple = (5, 10, 20),
                    n_eval_users: int = 200) -> dict:
    """
    Offline ranking evaluation.
    For each simulated user, ranks all items and computes nDCG/Hit@K.
    Relevant items = those with cosine sim above threshold.
    """
    rng = np.random.default_rng(42)
    results = {f"nDCG@{k}": [] for k in k_values}
    results.update({f"Hit@{k}": [] for k in k_values})

    n_users = min(n_eval_users, len(user_preferences))
    threshold = 0.45

    for uid in range(n_users):
        pref  = user_preferences[uid]
        sims  = item_embeddings @ pref            # (N,) cosine similarities
        ranked = np.argsort(-sims)                # descending
        relevant = set(np.where(sims > threshold)[0].tolist())

        if len(relevant) == 0:
            continue                               # skip users with no relevant items

        for k in k_values:
            results[f"nDCG@{k}"].append(ndcg_at_k(ranked.tolist(), relevant, k))
            results[f"Hit@{k}"].append(hit_at_k(ranked.tolist(),   relevant, k))

    summary = {metric: float(np.mean(vals)) for metric, vals in results.items()
               if vals}
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Slice / per-category analysis
# ─────────────────────────────────────────────────────────────────────────────

def slice_analysis(y_true: list[int], y_pred: list[int],
                   label_map: dict, texts: list[str] = None) -> pd.DataFrame:
    """
    Per-category F1, support, and sample failure cases.
    """
    report = classification_report(
        y_true, y_pred,
        labels=list(label_map.keys()),
        target_names=list(label_map.values()),
        output_dict=True, zero_division=0
    )
    rows = []
    for cat_name, metrics in report.items():
        if cat_name in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append({
            "category":  cat_name,
            "precision": round(metrics["precision"], 4),
            "recall":    round(metrics["recall"],    4),
            "f1":        round(metrics["f1-score"],  4),
            "support":   int(metrics["support"]),
        })
    df = pd.DataFrame(rows).sort_values("f1")
    return df


def error_analysis(y_true: list[int], y_pred: list[int],
                   texts: list[str], label_map: dict,
                   n_examples: int = 5) -> pd.DataFrame:
    """Return a sample of misclassified examples for manual inspection."""
    errors = [(t, p, txt)
              for t, p, txt in zip(y_true, y_pred, texts)
              if t != p]
    rng    = np.random.default_rng(42)
    sample = [errors[i] for i in rng.choice(len(errors),
              size=min(n_examples, len(errors)), replace=False)]
    rows   = [{"true":  label_map.get(t, t),
               "pred":  label_map.get(p, p),
               "text":  txt[:120] + "…"}
              for t, p, txt in sample]
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Confusion matrix plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_arr: np.ndarray, class_names: list[str],
                          save_path: Path):
    n = len(class_names)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.55)))

    # Normalise by true label counts
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_arr.astype(float), row_sums,
                         where=row_sums != 0, out=np.zeros_like(cm_arr, dtype=float))

    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_ylabel("True label")
    ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix (normalised)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[eval] Confusion matrix saved → {save_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Ablation study helper
# ─────────────────────────────────────────────────────────────────────────────

def ablation_study(item_embeddings_dict: dict,
                   agent_classes: list,
                   env_factory,
                   n_steps: int = 1000) -> pd.DataFrame:
    """
    Ablation 1: TF-IDF embeddings vs SBERT embeddings
    Ablation 2: ε-greedy vs LinUCB bandit

    Returns a DataFrame with mean reward for each combination.
    """
    import sys
    sys.path.insert(0, "src")
    from rl_agent import run_simulation

    rows = []
    for emb_name, embs in item_embeddings_dict.items():
        env = env_factory(embs)
        for AgentClass, agent_name, agent_kwargs in agent_classes:
            agent  = AgentClass(**agent_kwargs(len(embs), embs.shape[1]))
            result = run_simulation(env, agent, n_steps=n_steps)
            rows.append({
                "embedding":   emb_name,
                "agent":       agent_name,
                "mean_reward": round(result["mean_reward"], 4),
                "total_reward": round(result["total_reward"], 1),
            })
    df = pd.DataFrame(rows)
    print("\n[eval] Ablation results:")
    print(df.to_string(index=False))
    df.to_csv(RESULTS_DIR / "ablation_results.csv", index=False)
    return df


def plot_ablation(df: pd.DataFrame):
    """Bar chart for ablation results."""
    fig, ax = plt.subplots(figsize=(8, 4))
    labels  = [f"{r['embedding']}\n{r['agent']}" for _, r in df.iterrows()]
    values  = df["mean_reward"].tolist()
    colors  = ["steelblue" if "SBERT" in l else "darkorange" for l in labels]

    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_ylabel("Mean reward per step")
    ax.set_title("Ablation: Embedding type × Bandit algorithm")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "ablation_bar.png", dpi=120)
    plt.close()
    print(f"[eval] Ablation bar chart saved.")


# ─────────────────────────────────────────────────────────────────────────────
# Main evaluation runner
# ─────────────────────────────────────────────────────────────────────────────

def run_evaluation():
    """
    Load saved artefacts and run the full evaluation suite.
    Requires train.py to have been run first.
    """
    import sys
    sys.path.insert(0, "src")
    from rl_agent import (NewsRecommendationEnv, EpsilonGreedyBandit,
                          LinUCBBandit, run_simulation)

    print("\n" + "="*60)
    print("EVALUATION SUITE")
    print("="*60)

    # ── Load label map ──
    with open(DATA_DIR / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    # ── Load test split ──
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    test_df["clean_text"] = test_df["clean_text"].fillna("")

    # ── Load embeddings ──
    all_embs  = np.load(RESULTS_DIR / "article_embeddings.npy")
    test_embs = np.load(RESULTS_DIR / "test_embeddings.npy")
    dim       = all_embs.shape[1]
    print(f"  Loaded {len(all_embs):,} article embeddings (dim={dim})")

    # ── 1. Ranking metrics ──
    print("\n[1] Ranking Metrics (SBERT embeddings)")
    env_tmp  = NewsRecommendationEnv(all_embs, n_users=200)
    ranking  = evaluate_ranker(all_embs, env_tmp.user_prefs, k_values=(5,10,20))
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in ranking.items()))
    with open(RESULTS_DIR / "ranking_metrics.json", "w") as f:
        json.dump(ranking, f, indent=2)

    # ── 2. CNN evaluation (if model exists) ──
    cnn_metrics_path = RESULTS_DIR / "cnn_metrics.json"
    if cnn_metrics_path.exists():
        print("\n[2] CNN Classification Metrics")
        with open(cnn_metrics_path) as f:
            cm_data = json.load(f)
        print(f"  Accuracy : {cm_data['accuracy']:.4f}")
        print(f"  Macro-F1 : {cm_data['macro_f1']:.4f}")

        # Confusion matrix plot
        cm_arr = np.array(cm_data["conf_mat"])
        cat_names = [label_map[i] for i in sorted(label_map)]
        plot_confusion_matrix(cm_arr, cat_names,
                              RESULTS_DIR / "confusion_matrix.png")

    # ── 3. Slice analysis ──
    print("\n[3] Slice Analysis")
    if cnn_metrics_path.exists():
        cm_arr   = np.array(cm_data["conf_mat"])
        y_true   = []
        y_pred   = []
        for true_i, row in enumerate(cm_arr):
            for pred_i, count in enumerate(row):
                y_true.extend([true_i] * int(count))
                y_pred.extend([pred_i] * int(count))
        slice_df = slice_analysis(y_true, y_pred, label_map,
                                  texts=test_df["clean_text"].tolist())
        print(slice_df.to_string(index=False))
        slice_df.to_csv(RESULTS_DIR / "slice_analysis.csv", index=False)

    # ── 4. Ablation study ──
    print("\n[4] Ablation Study")

    # Ablation 1: SBERT vs reduced-dim TF-IDF (projected to 64-d)
    # For the ablation we project TF-IDF to match SBERT dim using SVD
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    tfidf_vec   = TfidfVectorizer(max_features=10_000)
    train_df    = pd.read_csv(DATA_DIR / "train.csv").fillna("")
    tfidf_mat   = tfidf_vec.fit_transform(train_df["clean_text"].tolist())
    svd         = TruncatedSVD(n_components=64, random_state=42)
    tfidf_proj  = svd.fit_transform(tfidf_mat).astype(np.float32)
    norms       = np.linalg.norm(tfidf_proj, axis=1, keepdims=True)
    tfidf_proj /= np.where(norms > 0, norms, 1)

    sbert_sub   = all_embs[:len(tfidf_proj)]   # same size subset

    emb_dict = {"SBERT": sbert_sub, "TF-IDF+SVD": tfidf_proj}

    def env_factory(embs):
        return NewsRecommendationEnv(embs, n_users=50, candidates_per_step=15)

    agent_classes = [
        (EpsilonGreedyBandit, "ε-greedy",
         lambda n, d: {"n_arms": n, "epsilon": 0.15}),
        (LinUCBBandit,        "LinUCB",
         lambda n, d: {"n_arms": n, "context_dim": d, "alpha": 0.5}),
    ]

    abl_df = ablation_study(emb_dict, agent_classes, env_factory, n_steps=1000)
    plot_ablation(abl_df)

    # ── 5. Error analysis ──
    print("\n[5] Error Analysis (sample of CNN misclassifications)")
    if cnn_metrics_path.exists():
        err_df = error_analysis(y_true[:200], y_pred[:200],
                                test_df["clean_text"].tolist()[:200],
                                label_map)
        print(err_df.to_string(index=False))
        err_df.to_csv(RESULTS_DIR / "error_analysis.csv", index=False)

    print(f"\n[eval] All results saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    run_evaluation()
