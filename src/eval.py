import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

from models.nlp_model import EmbeddingModel
from rl_agent import LinUCB
from metrics import ndcg_at_k, hit_at_k
from utils import save_json, plot, setup_dirs

def main():
    setup_dirs()

    # -------------------------
    # Load and preprocess dataset
    # -------------------------
    df = pd.read_json("data/News_Category_Dataset_v3.json", lines=True)
    df['text'] = df['headline'] + " " + df['short_description']
    df = df[['text', 'category']].dropna()

    # Encode categories as integers
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])

    # Split dataset
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])
    y_train = train['label'].values
    y_test = test['label'].values

    # -------------------------
    # Embedding model
    # -------------------------
    model = EmbeddingModel()
    model.fit(train['text'])
    X_train = model.transform(train['text'])
    X_test = model.transform(test['text'])

    # -------------------------
    # LinUCB agent
    # -------------------------
    n_categories = len(np.unique(y_train))
    agent = LinUCB(n_categories, X_train.shape[1])

    # Supervised "training" (assign reward = 1 for true label)
    for i in range(len(X_train)):
        context = X_train[i]
        action = y_train[i]
        reward = 1
        agent.update(action, context, reward)

    # -------------------------
    # Evaluation
    # -------------------------
    ndcg_scores = []
    hit_scores = []
    eval_records = []

    print(f"{'Step':>4} | {'True':>4} | {'Top1':>4} | {'Top2':>4} | {'Top3':>4} | {'nDCG@3':>7} | {'Hit@3':>6}")
    print("-"*60)

    for i in range(len(X_test)):
        context = X_test[i]
        scores = [context @ theta for theta in agent.thetas]
        ranked_labels = np.argsort(scores)[::-1]

        ndcg = ndcg_at_k(y_test[i], ranked_labels, k=3)
        hit = hit_at_k(y_test[i], ranked_labels, k=3)

        ndcg_scores.append(ndcg)
        hit_scores.append(hit)

        eval_records.append({
            "Step": i+1,
            "True_Label": int(y_test[i]),
            "Top1": int(ranked_labels[0]),
            "Top2": int(ranked_labels[1]),
            "Top3": int(ranked_labels[2]),
            "nDCG@3": ndcg,
            "Hit@3": hit
        })

        if (i+1) % 50 == 0:
            print(f"{i+1:>4} | {y_test[i]:>4} | {ranked_labels[0]:>4} | {ranked_labels[1]:>4} | {ranked_labels[2]:>4} | {ndcg:>7.3f} | {hit:>6}")

    # Save evaluation table
    eval_df = pd.DataFrame(eval_records)
    eval_df.to_csv("results/evaluation_table.csv", index=False)

    # Save metrics summary
    metrics = {
        "avg_nDCG@3": float(np.mean(ndcg_scores)),
        "avg_Hit@3": float(np.mean(hit_scores))
    }
    save_json(metrics, "results/metrics.json")

    # Plot curves
    plot(ndcg_scores, "results/ndcg_curve.png", "nDCG@3 Curve")
    plot(hit_scores, "results/hit_curve.png", "Hit@3 Curve")

    print("\nEvaluation complete!")
    print("Average nDCG@3:", metrics['avg_nDCG@3'])
    print("Average Hit@3:", metrics['avg_Hit@3'])
    print("Per-sample table saved: results/evaluation_table.csv")

if __name__ == "__main__":
    main()