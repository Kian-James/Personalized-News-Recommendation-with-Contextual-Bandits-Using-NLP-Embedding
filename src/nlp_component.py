"""
nlp_component.py
NLP pipeline for the news recommender.

Two modes:
  1. TF-IDF (fast, no GPU needed)          — used as the non-DL baseline
  2. Sentence-Transformers (all-MiniLM-L6)  — main NLP component for embeddings

Both expose the same interface: fit(texts) / transform(texts) → np.ndarray
"""

import numpy as np
import pickle
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MODELS_DIR = Path("experiments/results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# 1.  TF-IDF  (non-DL baseline)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFEmbedder:
    """
    Wraps sklearn TfidfVectorizer.
    Output shape: (N, vocab_size) — sparse → dense float32.
    """

    def __init__(self, max_features: int = 10_000, ngram_range=(1, 2)):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            min_df=3,
        )
        self.fitted = False

    def fit(self, texts: list[str]):
        self.vectorizer.fit(texts)
        self.fitted = True
        print(f"[TFIDFEmbedder] Fitted on {len(texts):,} texts. "
              f"Vocab size: {len(self.vectorizer.vocabulary_):,}")
        return self

    def transform(self, texts: list[str]) -> np.ndarray:
        assert self.fitted, "Call fit() first."
        return self.vectorizer.transform(texts).toarray().astype(np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    def save(self, path: Path = MODELS_DIR / "tfidf.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)
        print(f"[TFIDFEmbedder] Saved to {path}")

    def load(self, path: Path = MODELS_DIR / "tfidf.pkl"):
        with open(path, "rb") as f:
            self.vectorizer = pickle.load(f)
        self.fitted = True
        return self


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Sentence-Transformer  (main NLP component)
# ─────────────────────────────────────────────────────────────────────────────

class SentenceEmbedder:
    """
    Dense semantic embeddings via sentence-transformers (all-MiniLM-L6-v2).
    Output shape: (N, 384) — float32.

    This is the core NLP component satisfying the NLP requirement:
      - Pre-trained language model performing semantic text embedding
      - Enables downstream retrieval and ranking by cosine similarity
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self, model_name: str = MODEL_NAME, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.dim   = self.model.get_sentence_embedding_dimension()
        print(f"[SentenceEmbedder] Loaded '{model_name}', "
              f"embedding dim={self.dim}, device={device}")

    def transform(self, texts: list[str], batch_size: int = 256,
                  show_progress: bool = True) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=True,   # unit-norm → cosine = dot product
            convert_to_numpy=True,
        )
        return embeddings.astype(np.float32)

    # Alias so it matches TFIDFEmbedder interface
    def fit(self, texts):
        return self        # pre-trained; no fitting needed

    def fit_transform(self, texts: list[str], **kwargs) -> np.ndarray:
        return self.transform(texts, **kwargs)

    def save_embeddings(self, embeddings: np.ndarray,
                        path: Path = MODELS_DIR / "article_embeddings.npy"):
        np.save(path, embeddings)
        print(f"[SentenceEmbedder] Saved embeddings {embeddings.shape} → {path}")

    @staticmethod
    def load_embeddings(path: Path = MODELS_DIR / "article_embeddings.npy") -> np.ndarray:
        emb = np.load(path)
        print(f"[SentenceEmbedder] Loaded embeddings {emb.shape} from {path}")
        return emb


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Retrieval / ranking helpers
# ─────────────────────────────────────────────────────────────────────────────

def rank_by_similarity(query_vec: np.ndarray, item_vecs: np.ndarray,
                       top_k: int = 10) -> np.ndarray:
    """
    Return indices of top-k most similar items to the query vector.
    Both query_vec and item_vecs should be unit-normalised.
    """
    scores = item_vecs @ query_vec          # (N,) dot products = cosine sims
    top_k  = min(top_k, len(scores))
    return np.argpartition(scores, -top_k)[-top_k:][::-1]


def user_profile_vector(liked_indices: list[int],
                        item_vecs: np.ndarray) -> np.ndarray:
    """
    Aggregate user preferences into a single profile vector
    by averaging embeddings of liked items (then L2-normalising).
    """
    if not liked_indices:
        return np.zeros(item_vecs.shape[1], dtype=np.float32)
    profile = item_vecs[liked_indices].mean(axis=0)
    norm    = np.linalg.norm(profile)
    return profile / norm if norm > 0 else profile


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Quick smoke-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    sample_texts = [
        "Scientists discover new species of deep sea fish near Pacific trench",
        "Stock markets tumble as inflation fears rise across Europe",
        "New thriller novel tops bestseller list for third consecutive week",
        "Government announces major infrastructure spending bill",
        "Climate summit reaches landmark agreement on carbon emissions",
    ]

    print("=== TF-IDF Embedder ===")
    tfidf = TFIDFEmbedder(max_features=5000)
    vecs  = tfidf.fit_transform(sample_texts)
    print(f"  Shape: {vecs.shape}")

    print("\n=== Sentence Embedder ===")
    se    = SentenceEmbedder()
    svecs = se.fit_transform(sample_texts, show_progress=False)
    print(f"  Shape: {svecs.shape}")

    query = svecs[0]
    ranked = rank_by_similarity(query, svecs, top_k=3)
    print(f"\n  Most similar to '{sample_texts[0][:40]}...':")
    for idx in ranked:
        print(f"    [{idx}] {sample_texts[idx]}")
