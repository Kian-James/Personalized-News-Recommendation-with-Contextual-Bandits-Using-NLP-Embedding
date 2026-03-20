"""
train.py
Master training script for the News Recommender System.

Runs the full pipeline in order:
  1. Data loading & preprocessing
  2. NLP embeddings (TF-IDF baseline + Sentence-Transformers)
  3. TextCNN training (CNN component)
  4. RL bandit simulation (RL component)
  5. Save all artefacts

Usage:
    python src/train.py                     # full run
    python src/train.py --fast              # small subset for quick testing
    python src/train.py --skip-cnn          # skip CNN training
"""

import argparse
import json
import time
import numpy as np
import torch
from pathlib import Path

# ── project imports ───────────────────────────────────────────────────────────
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_pipeline  import run_pipeline, RAW_FILE
from nlp_component  import TFIDFEmbedder, SentenceEmbedder
from models.cnn_component import (
    Vocabulary, TextCNN, train_textcnn, evaluate_textcnn
)
from rl_agent       import (
    NewsRecommendationEnv, EpsilonGreedyBandit, LinUCBBandit,
    run_simulation, compare_agents, plot_learning_curves
)

RESULTS_DIR = Path("experiments/results")
MODELS_DIR  = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ─────────────────────────────────────────────────────────────────────────────
def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ─────────────────────────────────────────────────────────────────────────────
def step1_data(args):
    print("\n" + "="*60)
    print("STEP 1 — Data Pipeline")
    print("="*60)
    train, val, test, label_enc = run_pipeline(RAW_FILE)

    if args.fast:
        # Use a small subset for quick smoke-test
        train = train.sample(n=min(2000, len(train)), random_state=SEED)
        val   = val.sample(n=min(500,  len(val)),   random_state=SEED)
        test  = test.sample(n=min(500, len(test)),  random_state=SEED)
        print(f"[fast mode] Subset: train={len(train)} val={len(val)} test={len(test)}")

    return train, val, test, label_enc


# ─────────────────────────────────────────────────────────────────────────────
def step2_nlp(train, val, test, args):
    print("\n" + "="*60)
    print("STEP 2 — NLP Embeddings")
    print("="*60)

    # ── TF-IDF baseline ──
    print("\n[TF-IDF]")
    tfidf = TFIDFEmbedder(max_features=20_000)
    t0 = time.time()
    train_tfidf = tfidf.fit_transform(train["clean_text"].tolist())
    val_tfidf   = tfidf.transform(val["clean_text"].tolist())
    test_tfidf  = tfidf.transform(test["clean_text"].tolist())
    tfidf.save()
    print(f"  TF-IDF embeddings: {train_tfidf.shape}  ({time.time()-t0:.1f}s)")

    # ── Sentence-Transformers (main NLP component) ──
    print("\n[Sentence-Transformers: all-MiniLM-L6-v2]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    se = SentenceEmbedder(device=device)

    all_texts = (train["clean_text"].tolist() +
                 val["clean_text"].tolist() +
                 test["clean_text"].tolist())
    all_embs  = se.fit_transform(all_texts, batch_size=256)

    n_train = len(train)
    n_val   = len(val)
    train_embs = all_embs[:n_train]
    val_embs   = all_embs[n_train:n_train + n_val]
    test_embs  = all_embs[n_train + n_val:]

    se.save_embeddings(all_embs, RESULTS_DIR / "article_embeddings.npy")
    se.save_embeddings(train_embs, RESULTS_DIR / "train_embeddings.npy")
    se.save_embeddings(test_embs,  RESULTS_DIR / "test_embeddings.npy")

    return {
        "tfidf": (train_tfidf, val_tfidf, test_tfidf, tfidf),
        "sbert": (train_embs, val_embs, test_embs, all_embs),
    }


# ─────────────────────────────────────────────────────────────────────────────
def step3_cnn(train, val, test, label_enc, args):
    print("\n" + "="*60)
    print("STEP 3 — TextCNN (CNN Component)")
    print("="*60)

    if args.skip_cnn:
        print("  [--skip-cnn] Skipping CNN training.")
        return None, None

    device = get_device()
    print(f"  Device: {device}")

    # Build vocab from training set only
    vocab = Vocabulary(max_size=30_000, min_freq=2)
    vocab.build(train["clean_text"].tolist())

    num_classes = train["label"].nunique()

    model, history = train_textcnn(
        train_texts  = train["clean_text"].tolist(),
        train_labels = train["label"].tolist(),
        val_texts    = val["clean_text"].tolist(),
        val_labels   = val["label"].tolist(),
        vocab        = vocab,
        num_classes  = num_classes,
        epochs       = 5 if args.fast else 10,
        batch_size   = 128,
        lr           = 1e-3,
        device       = device,
    )

    # Evaluate on test set
    cnn_metrics = evaluate_textcnn(
        model, test["clean_text"].tolist(), test["label"].tolist(),
        vocab, device=device
    )
    print(f"\n[TextCNN Test Results]")
    print(f"  Accuracy  : {cnn_metrics['accuracy']:.4f}")
    print(f"  Macro-F1  : {cnn_metrics['macro_f1']:.4f}")

    # Save metrics
    cnn_save = {k: v.tolist() if hasattr(v, "tolist") else v
                for k, v in cnn_metrics.items()}
    with open(RESULTS_DIR / "cnn_metrics.json", "w") as f:
        json.dump(cnn_save, f, indent=2)

    # Save learning curves
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["train_loss"], marker="o")
    axes[0].set_title("TextCNN — Training Loss")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("CE Loss")
    axes[1].plot(history["val_acc"], marker="o", color="darkorange")
    axes[1].set_title("TextCNN — Validation Accuracy")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Accuracy")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "cnn_learning_curves.png", dpi=120)
    plt.close()
    print(f"  Learning curves saved.")

    return model, vocab


# ─────────────────────────────────────────────────────────────────────────────
def step4_rl(nlp_outputs, args):
    print("\n" + "="*60)
    print("STEP 4 — RL Contextual Bandit Simulation")
    print("="*60)

    _, _, _, all_embs = nlp_outputs["sbert"]

    n_steps = 500 if args.fast else 3000
    n_seeds = 1   if args.fast else 3

    env = NewsRecommendationEnv(
        item_embeddings     = all_embs,
        n_users             = 100,
        candidates_per_step = 20,
        click_threshold     = 0.50,
    )

    # Random baseline
    random_rew = env.random_policy_reward(n_steps=n_steps)
    print(f"  Random baseline mean reward: {random_rew:.4f}")

    # Run both agents
    results = compare_agents(env, n_steps=n_steps, n_seeds=n_seeds)

    # Detailed run for individual plots
    dim = all_embs.shape[1]
    for AgentClass, name, kwargs in [
        (EpsilonGreedyBandit, "EpsilonGreedy",
         {"n_arms": len(all_embs), "epsilon": 0.15}),
        (LinUCBBandit, "LinUCB",
         {"n_arms": len(all_embs), "context_dim": dim, "alpha": 0.5}),
    ]:
        agent = AgentClass(**kwargs)
        res   = run_simulation(env, agent, n_steps=n_steps)
        plot_learning_curves(
            res, agent_name=name,
            save_path=RESULTS_DIR / f"rl_{name.lower()}_curve.png"
        )
        print(f"  {name:18s} mean_reward={res['mean_reward']:.4f}  "
              f"total={res['total_reward']:.1f}")

    print(f"  RL results saved to {RESULTS_DIR}/")


# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train News Recommender System")
    parser.add_argument("--fast",     action="store_true",
                        help="Quick run with small data subset")
    parser.add_argument("--skip-cnn", action="store_true",
                        help="Skip TextCNN training")
    args = parser.parse_args()

    t_start = time.time()

    train, val, test, label_enc = step1_data(args)
    nlp_outputs = step2_nlp(train, val, test, args)
    cnn_model, vocab = step3_cnn(train, val, test, label_enc, args)
    step4_rl(nlp_outputs, args)

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Training complete in {elapsed/60:.1f} min.")
    print(f"All artefacts saved to {RESULTS_DIR}/")
    print("="*60)


if __name__ == "__main__":
    main()
