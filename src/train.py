import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from models.nlp_model import EmbeddingModel
from rl_agent import LinUCB
from utils import setup_dirs, save_json, plot

def main():
    setup_dirs()

    # Load Kaggle News dataset
    df = pd.read_json("data/News_Category_Dataset_v3.json", lines=True)

    # Use headline + short_description as text
    df['text'] = df['headline'] + " " + df['short_description']
    df = df[['text', 'category']].dropna()

    # Encode categories as integers
    le = LabelEncoder()
    df['label'] = le.fit_transform(df['category'])

    # Split dataset (simple stratified)
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Train embedding model
    model = EmbeddingModel()
    model.fit(train['text'])

    X = model.transform(test['text'])
    y = test['label'].values

    agent = LinUCB(len(set(y)), X.shape[1])

    rewards = []
    cum_rewards = []
    records = []
    total = 0

    print(f"{'Step':>4} | {'True':>4} | {'Action':>6} | {'Reward':>6} | {'Cumulative':>10}")
    print("-"*50)

    for i in range(len(X)):
        context = X[i]
        action = agent.select(context)
        reward = 1 if action == y[i] else 0
        agent.update(action, context, reward)

        total += reward
        rewards.append(reward)
        cum_rewards.append(total)

        records.append({
            "Step": i+1,
            "True_Label": int(y[i]),
            "Action": int(action),
            "Reward": reward,
            "Cumulative": total
        })

        # Print every 50 steps
        if (i+1) % 50 == 0:
            print(f"{i+1:>4} | {y[i]:>4} | {action:>6} | {reward:>6} | {total:>10}")

    # Save full training table
    df_records = pd.DataFrame(records)
    df_records.to_csv("results/training_table.csv", index=False)

    # Save logs + plot
    save_json({"rewards": rewards, "cumulative_reward": cum_rewards}, "logs/train.json")
    plot(cum_rewards, "results/reward_curve.png", "Cumulative Reward Curve")

    print(f"\nTraining Complete! Avg Reward: {np.mean(rewards):.4f}")
    print("Training table saved to results/training_table.csv")

if __name__ == "__main__":
    main()