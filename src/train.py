"""
train.py
Master training script — runs the full pipeline:
  1. Data pipeline
  2. NLP embeddings (TF-IDF baseline + SBERT)
  3a. EmbeddingCNN  — PRIMARY CNN (SBERT input, targets ~98% accuracy)
  3b. TextCNN       — FROM SCRATCH (ablation, satisfies spec requirement)
  3c. SBERT + LR    — strong non-DL baseline
  4. RL contextual bandit simulation

Each run saves to its own folder: experiments/results/run1/, run2/, etc.

Usage:
    python src/train.py            # full run
    python src/train.py --fast     # quick smoke-test
    python src/train.py --skip-cnn
"""

import argparse
import json
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline import run_pipeline, RAW_FILE
from nlp_component import TFIDFEmbedder, SentenceEmbedder
from models.cnn_component import (
    Vocabulary,
    EmbeddingCNN, train_embedding_cnn,
    TextCNN, train_textcnn,
    evaluate_model, evaluate_textcnn,
)
from rl_agent import (
    NewsRecommendationEnv, EpsilonGreedyBandit, LinUCBBandit,
    run_simulation, compare_agents, plot_learning_curves,
)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


def get_run_dir() -> Path:
    base = Path("experiments/results")
    base.mkdir(parents=True, exist_ok=True)
    existing = sorted([d for d in base.iterdir()
                       if d.is_dir() and d.name.startswith("run")])
    next_num = (int(existing[-1].name.replace("run", "")) + 1) if existing else 1
    run_dir  = base / f"run{next_num}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n[train] Results directory: {run_dir}/")
    return run_dir


def get_device():
    if torch.cuda.is_available():   return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ─── Step 1: Data ─────────────────────────────────────────────────────────────

def step1_data(args):
    print("\n" + "="*60)
    print("STEP 1 — Data Pipeline")
    print("="*60)
    train, val, test, label_enc = run_pipeline(RAW_FILE)
    if args.fast:
        train = train.sample(n=min(2000, len(train)), random_state=SEED)
        val   = val.sample(n=min(500,   len(val)),   random_state=SEED)
        test  = test.sample(n=min(500,  len(test)),  random_state=SEED)
        print(f"[fast] train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test, label_enc


# ─── Step 2: NLP Embeddings ───────────────────────────────────────────────────

def step2_nlp(train, val, test, run_dir, args):
    print("\n" + "="*60)
    print("STEP 2 — NLP Embeddings")
    print("="*60)

    print("\n[TF-IDF]")
    tfidf = TFIDFEmbedder(max_features=20_000)
    t0 = time.time()
    train_tfidf = tfidf.fit_transform(train["clean_text"].tolist())
    val_tfidf   = tfidf.transform(val["clean_text"].tolist())
    test_tfidf  = tfidf.transform(test["clean_text"].tolist())
    tfidf.save(run_dir / "tfidf.pkl")
    print(f"  Shape: {train_tfidf.shape}  ({time.time()-t0:.1f}s)")

    print("\n[SBERT: all-MiniLM-L6-v2]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    se = SentenceEmbedder(device=device)
    all_texts  = (train["clean_text"].tolist() +
                  val["clean_text"].tolist() +
                  test["clean_text"].tolist())
    all_embs   = se.fit_transform(all_texts, batch_size=256)
    n_tr, n_va = len(train), len(val)
    train_embs = all_embs[:n_tr]
    val_embs   = all_embs[n_tr:n_tr + n_va]
    test_embs  = all_embs[n_tr + n_va:]

    se.save_embeddings(all_embs,   run_dir / "article_embeddings.npy")
    se.save_embeddings(train_embs, run_dir / "train_embeddings.npy")
    se.save_embeddings(test_embs,  run_dir / "test_embeddings.npy")

    return {
        "tfidf": (train_tfidf, val_tfidf, test_tfidf, tfidf),
        "sbert": (train_embs, val_embs, test_embs, all_embs),
    }


# ─── Step 3a: EmbeddingCNN (PRIMARY — high accuracy) ─────────────────────────

def step3a_embedding_cnn(train, val, test, nlp_outputs, run_dir, args):
    print("\n" + "="*60)
    print("STEP 3a — EmbeddingCNN (SBERT input) — PRIMARY CNN")
    print("="*60)

    train_embs, val_embs, test_embs, _ = nlp_outputs["sbert"]
    num_classes = train["label"].nunique()
    device      = get_device()
    print(f"  Device: {device}  |  Classes: {num_classes}")

    model, history = train_embedding_cnn(
        train_embs  = train_embs,
        train_labels= train["label"].tolist(),
        val_embs    = val_embs,
        val_labels  = val["label"].tolist(),
        num_classes = num_classes,
        epochs      = 30 if args.fast else 200,
        batch_size  = 32,
        lr          = 5e-4,
        device      = device,
        save_path   = run_dir / "embedding_cnn_best.pt",
    )

    metrics = evaluate_model(model, test_embs, test["label"].tolist(),
                             device=device, use_embeddings=True)
    print(f"\n[EmbeddingCNN Test]  Accuracy={metrics['accuracy']:.4f}  "
          f"Macro-F1={metrics['macro_f1']:.4f}")

    save_metrics(metrics, history, run_dir, prefix="cnn")
    return model, history


# ─── Step 3b: TextCNN from scratch (ablation) ────────────────────────────────

def step3b_textcnn_scratch(train, val, test, run_dir, args):
    print("\n" + "="*60)
    print("STEP 3b — TextCNN from scratch (ablation)")
    print("="*60)

    device = get_device()
    vocab  = Vocabulary(max_size=30_000, min_freq=1)
    vocab.build(train["clean_text"].tolist())
    num_classes = train["label"].nunique()

    model, history = train_textcnn(
        train_texts  = train["clean_text"].tolist(),
        train_labels = train["label"].tolist(),
        val_texts    = val["clean_text"].tolist(),
        val_labels   = val["label"].tolist(),
        vocab        = vocab,
        num_classes  = num_classes,
        epochs       = 5 if args.fast else 15,
        batch_size   = 32,
        lr           = 1e-3,
        device       = device,
        save_path    = run_dir / "textcnn_scratch_best.pt",
    )

    metrics = evaluate_textcnn(model, test["clean_text"].tolist(),
                               test["label"].tolist(), vocab, device=device)
    print(f"\n[TextCNN Scratch Test]  Accuracy={metrics['accuracy']:.4f}  "
          f"Macro-F1={metrics['macro_f1']:.4f}")

    save_metrics(metrics, history, run_dir, prefix="textcnn_scratch")
    return model, vocab


# ─── Step 3c: SBERT + Logistic Regression (non-DL baseline) ──────────────────

def step3c_sbert_lr(train, val, test, nlp_outputs, run_dir):
    print("\n" + "="*60)
    print("STEP 3c — SBERT + Logistic Regression (non-DL baseline)")
    print("="*60)
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score

    train_embs, _, test_embs, _ = nlp_outputs["sbert"]
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42)
    clf.fit(train_embs, train["label"].tolist())

    preds    = clf.predict(test_embs)
    acc      = accuracy_score(test["label"].tolist(), preds)
    macro_f1 = f1_score(test["label"].tolist(), preds, average="macro")
    print(f"  Accuracy={acc:.4f}  Macro-F1={macro_f1:.4f}")

    with open(run_dir / "sbert_lr_metrics.json", "w") as f:
        json.dump({"accuracy": acc, "macro_f1": macro_f1}, f, indent=2)
    return clf


# ─── Step 4: RL ───────────────────────────────────────────────────────────────

def step4_rl(nlp_outputs, run_dir, args):
    print("\n" + "="*60)
    print("STEP 4 — RL Contextual Bandit Simulation")
    print("="*60)

    _, _, _, all_embs = nlp_outputs["sbert"]
    n_steps = 500  if args.fast else 3000
    n_seeds = 1    if args.fast else 3
    dim     = all_embs.shape[1]

    env = NewsRecommendationEnv(all_embs, n_users=100,
                                candidates_per_step=20, click_threshold=0.50)
    print(f"  Random baseline: {env.random_policy_reward(n_steps=n_steps):.4f}")

    compare_agents(env, n_steps=n_steps, n_seeds=n_seeds, save_dir=run_dir)

    for AgentClass, name, kwargs in [
        (EpsilonGreedyBandit, "EpsilonGreedy",
         {"n_arms": len(all_embs), "epsilon": 0.15}),
        (LinUCBBandit, "LinUCB",
         {"n_arms": len(all_embs), "context_dim": dim, "alpha": 0.5}),
    ]:
        agent = AgentClass(**kwargs)
        res   = run_simulation(env, agent, n_steps=n_steps)
        plot_learning_curves(res, agent_name=name,
                             save_path=run_dir / f"rl_{name.lower()}_curve.png")
        print(f"  {name:18s} mean={res['mean_reward']:.4f}  "
              f"total={res['total_reward']:.1f}")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def save_metrics(metrics, history, run_dir, prefix="cnn"):
    save = {k: v.tolist() if hasattr(v, "tolist") else v
            for k, v in metrics.items()}
    with open(run_dir / f"{prefix}_metrics.json", "w") as f:
        json.dump(save, f, indent=2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], marker="o")
    axes[0].set_title(f"{prefix} — Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[1].plot(history["val_acc"], marker="o", color="darkorange")
    axes[1].set_title(f"{prefix} — Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(run_dir / f"{prefix}_learning_curves.png", dpi=120)
    plt.close()


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fast",     action="store_true")
    parser.add_argument("--skip-cnn", action="store_true")
    args = parser.parse_args()

    run_dir = get_run_dir()
    t_start = time.time()

    with open(run_dir / "run_config.json", "w") as f:
        json.dump({"fast": args.fast, "seed": SEED,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}, f, indent=2)

    train, val, test, label_enc = step1_data(args)
    nlp_outputs = step2_nlp(train, val, test, run_dir, args)

    if not args.skip_cnn:
        step3a_embedding_cnn(train, val, test, nlp_outputs, run_dir, args)
        step3b_textcnn_scratch(train, val, test, run_dir, args)

    step3c_sbert_lr(train, val, test, nlp_outputs, run_dir)
    step4_rl(nlp_outputs, run_dir, args)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min  →  {run_dir}/")
    print("="*60)

    with open("experiments/results/.last_run", "w") as f:
        f.write(str(run_dir))


if __name__ == "__main__":
    main()