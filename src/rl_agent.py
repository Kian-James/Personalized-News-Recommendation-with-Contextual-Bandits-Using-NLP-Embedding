"""
rl_agent.py
Contextual Bandit agents for the news recommender.

Two agents are implemented:
  1. EpsilonGreedyBandit  — simple ε-greedy exploration
  2. LinUCBBandit         — linear upper confidence bound (Disjoint model)

Both follow the same interface:
  select(context, candidate_indices)  →  chosen_index
  update(chosen_index, reward, context)

Environment spec
────────────────
  State   : context vector c ∈ R^d (user profile embedding at time t)
  Action  : index of one article to recommend from the candidate pool
  Reward  : +1 if simulated user clicks, 0 otherwise
  Episode : T steps per simulated user session

Reward function
───────────────
  R(a, u) = 1  if cosine_sim(item_emb[a], user_pref_emb[u]) > threshold
            else Bernoulli(0.05)   (small random click probability)
"""

import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path

RESULTS_DIR = Path("experiments/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(42)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  ε-greedy Bandit
# ─────────────────────────────────────────────────────────────────────────────

class EpsilonGreedyBandit:
    """
    Contextual ε-greedy bandit.
    Maintains a per-action running average reward estimate.
    Exploration decays over time: ε_t = ε_0 / (1 + decay * t)
    """

    def __init__(self, n_arms: int, epsilon: float = 0.15,
                 decay: float = 0.001):
        self.n_arms   = n_arms
        self.epsilon  = epsilon
        self.decay    = decay
        self.counts   = np.zeros(n_arms)          # pulls per arm
        self.values   = np.zeros(n_arms)          # mean reward per arm
        self.t        = 0
        self.history  = []                        # (step, reward) log

    def _current_epsilon(self) -> float:
        return self.epsilon / (1.0 + self.decay * self.t)

    def select(self, context: np.ndarray,
               candidate_indices: np.ndarray) -> int:
        """
        context           : user profile vector (not used by ε-greedy,
                            kept for interface compatibility)
        candidate_indices : subset of arm indices to choose from
        """
        eps = self._current_epsilon()
        if RNG.random() < eps:
            return int(RNG.choice(candidate_indices))   # explore
        # Exploit: pick arm with highest estimated value among candidates
        vals = self.values[candidate_indices]
        return int(candidate_indices[np.argmax(vals)])

    def update(self, arm: int, reward: float, context: np.ndarray = None):
        self.t          += 1
        self.counts[arm] += 1
        n                = self.counts[arm]
        self.values[arm] += (reward - self.values[arm]) / n
        self.history.append((self.t, reward))

    def cumulative_reward(self) -> np.ndarray:
        return np.cumsum([r for _, r in self.history])

    def reset(self):
        self.counts  = np.zeros(self.n_arms)
        self.values  = np.zeros(self.n_arms)
        self.t       = 0
        self.history = []


# ─────────────────────────────────────────────────────────────────────────────
# 2.  LinUCB Bandit (Disjoint model)
# ─────────────────────────────────────────────────────────────────────────────

class LinUCBBandit:
    """
    LinUCB with disjoint linear models (Li et al. 2010).

    For each arm a, maintains:
      A_a  : d×d matrix  (initialised to I_d)
      b_a  : d-vector    (initialised to 0)

    The UCB score for arm a given context x is:
      θ_a = A_a^{-1} b_a
      score = θ_a·x + α * sqrt(x·A_a^{-1}·x)

    α controls the exploration–exploitation tradeoff.
    """

    def __init__(self, n_arms: int, context_dim: int, alpha: float = 0.5):
        self.n_arms     = n_arms
        self.d          = context_dim
        self.alpha      = alpha
        self.A          = np.array([np.eye(context_dim)] * n_arms)  # (K, d, d)
        self.b          = np.zeros((n_arms, context_dim))            # (K, d)
        self.t          = 0
        self.history    = []

    def select(self, context: np.ndarray,
               candidate_indices: np.ndarray) -> int:
        x     = context.astype(np.float64)
        scores = []
        for a in candidate_indices:
            A_inv = np.linalg.solve(self.A[a], np.eye(self.d))
            theta = A_inv @ self.b[a]
            ucb   = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
            scores.append(ucb)
        best_local = int(np.argmax(scores))
        return int(candidate_indices[best_local])

    def update(self, arm: int, reward: float, context: np.ndarray):
        x            = context.astype(np.float64)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.t      += 1
        self.history.append((self.t, reward))

    def cumulative_reward(self) -> np.ndarray:
        return np.cumsum([r for _, r in self.history])

    def reset(self):
        self.A = np.array([np.eye(self.d)] * self.n_arms)
        self.b = np.zeros((self.n_arms, self.d))
        self.t       = 0
        self.history = []


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Offline simulation environment
# ─────────────────────────────────────────────────────────────────────────────

class NewsRecommendationEnv:
    """
    Offline simulation of a user interacting with the recommender.

    Simulated users each have a latent preference vector drawn from
    the item embedding space.  At each step:
      1. The agent selects one article from a random candidate pool.
      2. The environment simulates a click using cosine similarity
         between the article embedding and the user's preference.
    """

    def __init__(self, item_embeddings: np.ndarray,
                 n_users: int         = 200,
                 candidates_per_step: int = 20,
                 click_threshold: float   = 0.50,
                 noise_prob: float        = 0.05):
        self.embs              = item_embeddings          # (N, d)
        self.n_items           = len(item_embeddings)
        self.dim               = item_embeddings.shape[1]
        self.n_users           = n_users
        self.k                 = candidates_per_step
        self.click_threshold   = click_threshold
        self.noise_prob        = noise_prob

        # Synthesise user preference vectors as random combos of item embeddings
        rng = np.random.default_rng(42)
        indices = rng.integers(0, self.n_items, size=(n_users, 5))
        prefs   = self.embs[indices].mean(axis=1)
        norms   = np.linalg.norm(prefs, axis=1, keepdims=True)
        self.user_prefs = prefs / np.where(norms > 0, norms, 1)  # (U, d)

    def step(self, user_id: int, chosen_arm: int) -> float:
        """Compute reward for recommending article `chosen_arm` to `user_id`."""
        sim    = float(self.embs[chosen_arm] @ self.user_prefs[user_id])
        click  = sim > self.click_threshold
        # Small base click prob to avoid zero reward early on
        if not click:
            click = RNG.random() < self.noise_prob
        return float(click)

    def candidate_pool(self, user_id: int) -> np.ndarray:
        """Return random candidate item indices for a given user."""
        return RNG.choice(self.n_items, size=self.k, replace=False)

    def user_context(self, user_id: int) -> np.ndarray:
        """Return the (simulated) context vector for a user."""
        return self.user_prefs[user_id]

    def random_policy_reward(self, n_steps: int = 1000) -> float:
        """Baseline: purely random recommendations."""
        total = 0.0
        for _ in range(n_steps):
            uid   = int(RNG.integers(0, self.n_users))
            pool  = self.candidate_pool(uid)
            arm   = int(RNG.choice(pool))
            total += self.step(uid, arm)
        return total / n_steps


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Run simulation & plot learning curves
# ─────────────────────────────────────────────────────────────────────────────

def run_simulation(env: NewsRecommendationEnv,
                   agent,
                   n_steps: int    = 2000,
                   window: int     = 50) -> dict:
    """
    Run agent for n_steps in the environment.
    Returns dict with cumulative reward and rolling avg reward.
    """
    rewards = []
    for step in range(n_steps):
        uid    = int(RNG.integers(0, env.n_users))
        pool   = env.candidate_pool(uid)
        ctx    = env.user_context(uid)
        arm    = agent.select(ctx, pool)
        reward = env.step(uid, arm)
        agent.update(arm, reward, ctx)
        rewards.append(reward)

    rewards_arr = np.array(rewards)
    cumulative  = np.cumsum(rewards_arr)
    rolling     = np.convolve(rewards_arr, np.ones(window)/window, mode="valid")

    return {
        "rewards":    rewards_arr,
        "cumulative": cumulative,
        "rolling":    rolling,
        "mean_reward": float(rewards_arr.mean()),
        "total_reward": float(cumulative[-1]),
    }


def plot_learning_curves(results: dict, agent_name: str,
                         save_path: Path = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(results["cumulative"], color="steelblue")
    axes[0].set_title(f"{agent_name} — Cumulative Reward")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Cumulative reward")

    axes[1].plot(results["rolling"], color="darkorange")
    axes[1].axhline(results["mean_reward"], color="gray",
                    linestyle="--", label=f"Mean={results['mean_reward']:.3f}")
    axes[1].set_title(f"{agent_name} — Rolling Avg Reward")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Avg reward (rolling)")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        print(f"[rl_agent] Saved learning curve → {save_path}")
    plt.close()


def compare_agents(env: NewsRecommendationEnv, n_steps: int = 2000,
                   n_seeds: int = 3):
    """Run all agents across multiple seeds and compare."""
    agent_results = {}
    dim = env.dim

    for seed in range(n_seeds):
        rng_seed = 100 * seed
        eg  = EpsilonGreedyBandit(env.n_items, epsilon=0.15)
        ucb = LinUCBBandit(env.n_items, context_dim=dim, alpha=0.5)

        for name, agent in [("EpsilonGreedy", eg), ("LinUCB", ucb)]:
            agent.reset()
            res = run_simulation(env, agent, n_steps=n_steps)
            agent_results.setdefault(name, []).append(res["rewards"])

    # Average across seeds
    summary = {}
    for name, seed_rewards in agent_results.items():
        arr = np.array(seed_rewards)          # (n_seeds, n_steps)
        summary[name] = {
            "mean":   arr.mean(axis=0),
            "std":    arr.std(axis=0),
            "cumulative": arr.cumsum(axis=1).mean(axis=0),
        }

    # Random baseline
    random_reward = env.random_policy_reward(n_steps=n_steps)
    summary["Random"] = {"mean_scalar": random_reward}

    # Plot comparison
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"EpsilonGreedy": "steelblue", "LinUCB": "darkorange"}
    window = 50
    for name, data in summary.items():
        if name == "Random":
            ax.axhline(data["mean_scalar"], color="gray",
                       linestyle="--", label=f"Random ({data['mean_scalar']:.3f})")
            continue
        rolling = np.convolve(data["mean"], np.ones(window)/window, mode="valid")
        ax.plot(rolling, label=name, color=colors[name])

    ax.set_title("Agent Comparison — Rolling Avg Reward (mean over seeds)")
    ax.set_xlabel("Step")
    ax.set_ylabel(f"Avg reward (window={window})")
    ax.legend()
    plt.tight_layout()
    save_p = RESULTS_DIR / "rl_comparison.png"
    plt.savefig(save_p, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[rl_agent] Comparison plot saved → {save_p}")

    # Save numeric summary
    numeric = {k: {"total_cumulative": float(v["cumulative"][-1])}
               for k, v in summary.items() if k != "Random"}
    numeric["Random"] = {"mean_reward": float(random_reward)}
    with open(RESULTS_DIR / "rl_summary.json", "w") as f:
        json.dump(numeric, f, indent=2)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Dummy embeddings
    N, D = 500, 64
    dummy_embs = np.random.randn(N, D).astype(np.float32)
    dummy_embs /= np.linalg.norm(dummy_embs, axis=1, keepdims=True)

    env = NewsRecommendationEnv(dummy_embs, n_users=50, candidates_per_step=10)

    eg  = EpsilonGreedyBandit(N, epsilon=0.15)
    res = run_simulation(env, eg, n_steps=500)
    print(f"EpsilonGreedy — mean reward: {res['mean_reward']:.4f}")

    ucb = LinUCBBandit(N, context_dim=D, alpha=0.5)
    res = run_simulation(env, ucb, n_steps=500)
    print(f"LinUCB       — mean reward: {res['mean_reward']:.4f}")
