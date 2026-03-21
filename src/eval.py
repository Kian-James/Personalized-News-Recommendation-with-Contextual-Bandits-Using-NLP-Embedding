"""
eval.py
Evaluation suite — reads from the last run directory automatically,
or pass --run runN to evaluate a specific run.

Usage:
    python src/eval.py              # evaluates latest run
    python src/eval.py --run run2   # evaluates run2
"""

import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix, classification_report
)

DATA_DIR = Path("data/processed")


def get_run_dir(run_name: str = None) -> Path:
    base = Path("experiments/results")
    if run_name:
        run_dir = base / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir
    # Auto-detect last run
    last_run_file = base / ".last_run"
    if last_run_file.exists():
        return Path(last_run_file.read_text().strip())
    # Fallback: pick the highest-numbered run folder
    existing = sorted([d for d in base.iterdir()
                       if d.is_dir() and d.name.startswith("run")])
    if not existing:
        raise FileNotFoundError("No run folders found in experiments/results/")
    return existing[-1]


# ── Ranking metrics ───────────────────────────────────────────────────────────

def dcg_at_k(relevances, k):
    rels = np.asarray(relevances[:k], dtype=float)
    if len(rels) == 0:
        return 0.0
    return float((rels / np.log2(np.arange(2, len(rels) + 2))).sum())

def ndcg_at_k(recommended, relevant, k):
    rel   = [1.0 if i in relevant else 0.0 for i in recommended[:k]]
    ideal = sorted(rel, reverse=True)
    dcg   = dcg_at_k(rel, k)
    idcg  = dcg_at_k(ideal, k)
    return dcg / idcg if idcg > 0 else 0.0

def hit_at_k(recommended, relevant, k):
    return float(any(i in relevant for i in recommended[:k]))

def evaluate_ranker(train_embeddings, train_labels,
                    test_embeddings, test_labels,
                    k_values=(5, 10, 20)):
    # Honest offline ranking: each test article is a query against the
    # deduplicated train corpus. Relevant = same-category train articles.
    # Query is never in the corpus (train/test split), so model must
    # retrieve by semantic similarity, not exact match.
    results          = {f"nDCG@{k}": [] for k in k_values}
    results.update({f"Hit@{k}": [] for k in k_values})
    train_labels_arr = np.array(train_labels)

    for i in range(len(test_embeddings)):
        query    = test_embeddings[i]
        true_cat = test_labels[i]
        relevant = set(np.where(train_labels_arr == true_cat)[0].tolist())
        if not relevant:
            continue
        sims   = train_embeddings @ query
        ranked = np.argsort(-sims).tolist()
        for k in k_values:
            results[f"nDCG@{k}"].append(ndcg_at_k(ranked, relevant, k))
            results[f"Hit@{k}"].append(hit_at_k(ranked,   relevant, k))

    return {m: float(np.mean(v)) for m, v in results.items() if v}


# ── Slice analysis ────────────────────────────────────────────────────────────

def slice_analysis(y_true, y_pred, label_map):
    report = classification_report(
        y_true, y_pred,
        labels=list(label_map.keys()),
        target_names=list(label_map.values()),
        output_dict=True, zero_division=0
    )
    rows = []
    for name, metrics in report.items():
        if name in ("accuracy", "macro avg", "weighted avg"):
            continue
        rows.append({"category": name,
                     "precision": round(metrics["precision"], 4),
                     "recall":    round(metrics["recall"],    4),
                     "f1":        round(metrics["f1-score"],  4),
                     "support":   int(metrics["support"])})
    return pd.DataFrame(rows).sort_values("f1")


def error_analysis(y_true, y_pred, texts, label_map, n_examples=5):
    errors = [(t, p, txt) for t, p, txt in zip(y_true, y_pred, texts) if t != p]
    if not errors:
        return pd.DataFrame()
    rng    = np.random.default_rng(42)
    sample = [errors[i] for i in rng.choice(len(errors),
              size=min(n_examples, len(errors)), replace=False)]
    return pd.DataFrame([{"true": label_map.get(t, t),
                          "pred": label_map.get(p, p),
                          "text": txt[:120] + "…"}
                         for t, p, txt in sample])


# ── Confusion matrix ──────────────────────────────────────────────────────────

def plot_confusion_matrix(cm_arr, class_names, save_path):
    n        = len(class_names)
    fig, ax  = plt.subplots(figsize=(max(8, n * 0.6), max(6, n * 0.55)))
    row_sums = cm_arr.sum(axis=1, keepdims=True)
    cm_norm  = np.divide(cm_arr.astype(float), row_sums,
                         where=row_sums != 0, out=np.zeros_like(cm_arr, dtype=float))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap=cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ticks = np.arange(n)
    ax.set_xticks(ticks); ax.set_yticks(ticks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    ax.set_ylabel("True label"); ax.set_xlabel("Predicted label")
    ax.set_title("Confusion Matrix (normalised)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[eval] Confusion matrix saved → {save_path}")


# ── Ablation ──────────────────────────────────────────────────────────────────

def ablation_study(item_embeddings_dict, agent_classes, env_factory,
                   n_steps=1000, save_dir=None):
    import sys; sys.path.insert(0, "src")
    from rl_agent import run_simulation

    rows = []
    for emb_name, embs in item_embeddings_dict.items():
        env = env_factory(embs)
        for AgentClass, agent_name, agent_kwargs in agent_classes:
            agent  = AgentClass(**agent_kwargs(len(embs), embs.shape[1]))
            result = run_simulation(env, agent, n_steps=n_steps)
            rows.append({"embedding":    emb_name,
                         "agent":        agent_name,
                         "mean_reward":  round(result["mean_reward"],  4),
                         "total_reward": round(result["total_reward"], 1)})
    df = pd.DataFrame(rows)
    print("\n[eval] Ablation results:")
    print(df.to_string(index=False))
    if save_dir:
        df.to_csv(save_dir / "ablation_results.csv", index=False)
        _plot_ablation(df, save_dir)
    return df

def _plot_ablation(df, save_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    labels  = [f"{r['embedding']}\n{r['agent']}" for _, r in df.iterrows()]
    values  = df["mean_reward"].tolist()
    colors  = ["steelblue" if "SBERT" in l else "darkorange" for l in labels]
    bars    = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_ylim(0, max(values) * 1.2)
    ax.set_ylabel("Mean reward per step")
    ax.set_title("Ablation: Embedding type x Bandit algorithm")
    plt.tight_layout()
    plt.savefig(save_dir / "ablation_bar.png", dpi=120)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run_evaluation(run_dir: Path):
    import sys; sys.path.insert(0, "src")
    from rl_agent import (NewsRecommendationEnv, EpsilonGreedyBandit,
                          LinUCBBandit, run_simulation)

    print("\n" + "="*60)
    print(f"EVALUATION SUITE  —  {run_dir.name}")
    print("="*60)

    with open(DATA_DIR / "label_map.json") as f:
        label_map = {int(k): v for k, v in json.load(f).items()}

    test_df  = pd.read_csv(DATA_DIR / "test.csv").fillna("")
    all_embs = np.load(run_dir / "article_embeddings.npy")
    dim      = all_embs.shape[1]
    print(f"  Loaded {len(all_embs):,} embeddings (dim={dim}) from {run_dir.name}")

    # 1. Ranking — use test articles as queries, train as corpus
    # Use val set as the query corpus (never seen during training)
    # and deduplicated train labels to avoid inflated scores from oversampling
    print("\n[1] Ranking Metrics")
    train_df   = pd.read_csv(DATA_DIR / "train.csv").fillna("")
    val_df     = pd.read_csv(DATA_DIR / "val.csv").fillna("")
    train_embs_full = np.load(run_dir / "train_embeddings.npy")
    test_embs_rank  = np.load(run_dir / "test_embeddings.npy")

    # Deduplicate: train CSV may have oversampled rows; keep only unique ones
    unique_mask = ~train_df.duplicated(subset=["clean_text"])
    train_embs_unique  = train_embs_full[unique_mask.values]
    train_labels_unique = train_df[unique_mask]["label"].tolist()
    test_labels_rank   = test_df["label"].tolist()

    print(f"  Corpus: {len(train_embs_unique):,} unique train articles  "
          f"| Queries: {len(test_embs_rank):,} test articles")

    ranking = evaluate_ranker(
        train_embs_unique, train_labels_unique,
        test_embs_rank, test_labels_rank,
        k_values=(5, 10, 20)
    )
    print("  " + "  ".join(f"{k}={v:.4f}" for k, v in ranking.items()))
    with open(run_dir / "ranking_metrics.json", "w") as f:
        json.dump(ranking, f, indent=2)

    # 2. CNN — re-run inference on test set for honest metrics
    cnn_path   = run_dir / "cnn_metrics.json"
    model_path = run_dir / "embedding_cnn_best.pt"
    test_embs  = np.load(run_dir / "test_embeddings.npy")
    y_true     = test_df["label"].tolist()
    y_pred     = []

    if model_path.exists():
        print("\n[2] CNN Classification Metrics  (live inference on test set)")
        import torch as _torch, sys as _sys
        _sys.path.insert(0, "src")
        from models.cnn_component import EmbeddingCNN, EmbeddingDataset
        _num_cls = len(label_map)
        _model   = EmbeddingCNN(embed_dim=test_embs.shape[1], num_classes=_num_cls)
        _model.load_state_dict(_torch.load(model_path, map_location="cpu",
                                           weights_only=True))
        _model.eval()
        _ds = EmbeddingDataset(test_embs, y_true)
        _dl = _torch.utils.data.DataLoader(_ds, batch_size=256)
        with _torch.no_grad():
            for _xb, _ in _dl:
                y_pred.extend(_model(_xb).argmax(1).numpy().tolist())
        from sklearn.metrics import (accuracy_score as _acc,
                                     f1_score as _f1,
                                     confusion_matrix as _cm)
        acc = _acc(y_true, y_pred)
        mf1 = _f1(y_true, y_pred, average="macro")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Macro-F1 : {mf1:.4f}")
        with open(cnn_path, "w") as f:
            json.dump({"accuracy": acc, "macro_f1": mf1,
                       "conf_mat": _cm(y_true, y_pred).tolist()}, f, indent=2)
        cm_arr    = _cm(y_true, y_pred)
        cat_names = [label_map[i] for i in sorted(label_map)]
        plot_confusion_matrix(cm_arr, cat_names, run_dir / "confusion_matrix.png")

    elif cnn_path.exists():
        print("\n[2] CNN (from saved metrics)")
        with open(cnn_path) as f:
            cm_data = json.load(f)
        print(f"  Accuracy : {cm_data['accuracy']:.4f}")
        print(f"  Macro-F1 : {cm_data['macro_f1']:.4f}")
        cm_arr    = np.array(cm_data["conf_mat"])
        cat_names = [label_map[i] for i in sorted(label_map)]
        plot_confusion_matrix(cm_arr, cat_names, run_dir / "confusion_matrix.png")
        for true_i, row in enumerate(cm_arr):
            for pred_i, count in enumerate(row):
                y_true.extend([true_i] * int(count))
                y_pred.extend([pred_i] * int(count))

    # 3. Slice
    print("\n[3] Slice Analysis")
    if y_true:
        slice_df = slice_analysis(y_true, y_pred, label_map)
        print(slice_df.to_string(index=False))
        slice_df.to_csv(run_dir / "slice_analysis.csv", index=False)

    # 4. Ablation
    print("\n[4] Ablation Study")
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    train_df   = pd.read_csv(DATA_DIR / "train.csv").fillna("")
    tfidf_vec  = TfidfVectorizer(max_features=10_000)
    tfidf_mat  = tfidf_vec.fit_transform(train_df["clean_text"].tolist())
    svd        = TruncatedSVD(n_components=64, random_state=42)
    tfidf_proj = svd.fit_transform(tfidf_mat).astype(np.float32)
    norms      = np.linalg.norm(tfidf_proj, axis=1, keepdims=True)
    tfidf_proj /= np.where(norms > 0, norms, 1)
    sbert_sub   = all_embs[:len(tfidf_proj)]

    emb_dict = {"SBERT": sbert_sub, "TF-IDF+SVD": tfidf_proj}
    env_factory   = lambda embs: NewsRecommendationEnv(embs, n_users=50, candidates_per_step=15)
    agent_classes = [
        (EpsilonGreedyBandit, "ε-greedy", lambda n, d: {"n_arms": n, "epsilon": 0.15}),
        (LinUCBBandit,        "LinUCB",   lambda n, d: {"n_arms": n, "context_dim": d, "alpha": 0.5}),
    ]
    ablation_study(emb_dict, agent_classes, env_factory, n_steps=1000, save_dir=run_dir)

    # 5. Error analysis
    print("\n[5] Error Analysis")
    if y_true:
        err_df = error_analysis(y_true[:200], y_pred[:200],
                                test_df["clean_text"].tolist()[:200], label_map)
        if not err_df.empty:
            print(err_df.to_string(index=False))
            err_df.to_csv(run_dir / "error_analysis.csv", index=False)
        else:
            print("  No misclassifications found.")

    print(f"\n[eval] All results saved to {run_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=str, default=None,
                        help="Run folder name e.g. run2 (default: latest run)")
    args = parser.parse_args()

    run_dir = get_run_dir(args.run)
    run_evaluation(run_dir)