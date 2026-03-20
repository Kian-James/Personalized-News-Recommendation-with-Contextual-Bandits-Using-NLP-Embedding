"""
cnn_component.py
TextCNN for news category classification.

Architecture (Kim 2014 style):
  Embedding → Parallel Conv1d (filter sizes 2,3,4) → MaxPool → Dropout → FC

This satisfies the CNN requirement:
  - Convolutional layers extract local n-gram features from text
  - Used both as a standalone classifier and to produce category-aware
    feature vectors that augment the item embeddings for the ranker.

Also includes a small CNN built from scratch (no pre-trained weights)
to satisfy the "at least one model from scratch" requirement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

MODELS_DIR = Path("experiments/results")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary builder
# ─────────────────────────────────────────────────────────────────────────────

class Vocabulary:
    PAD = "<PAD>"
    UNK = "<UNK>"

    def __init__(self, max_size: int = 30_000, min_freq: int = 2):
        self.max_size  = max_size
        self.min_freq  = min_freq
        self.word2idx  = {self.PAD: 0, self.UNK: 1}
        self.idx2word  = {0: self.PAD, 1: self.UNK}

    def build(self, texts: list[str]):
        from collections import Counter
        counter = Counter()
        for text in texts:
            counter.update(text.split())
        common = [w for w, c in counter.most_common(self.max_size - 2)
                  if c >= self.min_freq]
        for w in common:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w
        print(f"[Vocabulary] Built vocab size={len(self.word2idx):,}")
        return self

    def encode(self, text: str, max_len: int = 50) -> list[int]:
        tokens = text.split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        # Pad / truncate to max_len
        ids   += [0] * (max_len - len(ids))
        return ids

    def __len__(self):
        return len(self.word2idx)


# ─────────────────────────────────────────────────────────────────────────────
# TextCNN model
# ─────────────────────────────────────────────────────────────────────────────

class TextCNN(nn.Module):
    """
    Multi-filter-size TextCNN.
    Built entirely from scratch (no pre-trained weights).
    """

    def __init__(self,
                 vocab_size: int,
                 embed_dim: int      = 128,
                 num_classes: int    = 20,
                 filter_sizes: tuple = (2, 3, 4),
                 num_filters: int    = 100,
                 dropout: float      = 0.5,
                 pad_idx: int        = 0):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)

        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=num_filters,
                      kernel_size=fs)
            for fs in filter_sizes
        ])

        self.dropout = nn.Dropout(dropout)
        self.fc      = nn.Linear(num_filters * len(filter_sizes), num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc.weight)
        for conv in self.convs:
            nn.init.kaiming_normal_(conv.weight)

    def forward(self, x: torch.Tensor,
                return_features: bool = False) -> torch.Tensor:
        """
        x: (batch, seq_len) int tensor
        Returns logits (batch, num_classes) or feature vector if return_features=True.
        """
        emb = self.embedding(x)          # (B, L, E)
        emb = emb.permute(0, 2, 1)       # (B, E, L) — Conv1d expects (B, C, L)

        pooled = []
        for conv in self.convs:
            c = F.relu(conv(emb))        # (B, num_filters, L - fs + 1)
            p = c.max(dim=2).values      # (B, num_filters) — max-over-time pooling
            pooled.append(p)

        cat = torch.cat(pooled, dim=1)   # (B, num_filters * len(filter_sizes))
        cat = self.dropout(cat)

        if return_features:
            return cat                   # raw feature vector for downstream use

        return self.fc(cat)              # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Dataset wrapper for PyTorch
# ─────────────────────────────────────────────────────────────────────────────

class NewsDataset(torch.utils.data.Dataset):
    def __init__(self, texts: list[str], labels: list[int],
                 vocab: Vocabulary, max_len: int = 50):
        self.ids    = [vocab.encode(t, max_len) for t in texts]
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (torch.tensor(self.ids[idx], dtype=torch.long),
                torch.tensor(self.labels[idx], dtype=torch.long))


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_textcnn(train_texts, train_labels, val_texts, val_labels,
                  vocab: Vocabulary,
                  num_classes: int,
                  epochs: int     = 10,
                  batch_size: int = 128,
                  lr: float       = 1e-3,
                  max_len: int    = 50,
                  device: str     = "cpu") -> TextCNN:

    torch.manual_seed(42)

    train_ds  = NewsDataset(train_texts, train_labels, vocab, max_len)
    val_ds    = NewsDataset(val_texts,   val_labels,   vocab, max_len)
    train_dl  = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_dl    = torch.utils.data.DataLoader(val_ds,   batch_size=256)

    model     = TextCNN(len(vocab), num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    history   = {"train_loss": [], "val_acc": []}
    best_acc  = 0.0
    patience  = 3
    no_improve = 0

    for epoch in range(1, epochs + 1):
        # ── train ──
        model.train()
        total_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(xb)
        avg_loss = total_loss / len(train_ds)

        # ── validate ──
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                preds   = model(xb).argmax(1)
                correct += (preds == yb).sum().item()
                total   += len(yb)
        val_acc = correct / total

        history["train_loss"].append(avg_loss)
        history["val_acc"].append(val_acc)
        print(f"  Epoch {epoch:2d}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        # Early stopping
        if val_acc > best_acc:
            best_acc  = val_acc
            no_improve = 0
            torch.save(model.state_dict(), MODELS_DIR / "textcnn_best.pt")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping at epoch {epoch}.")
                break

        scheduler.step()

    # Reload best weights
    model.load_state_dict(torch.load(MODELS_DIR / "textcnn_best.pt",
                                     map_location=device))
    print(f"[TextCNN] Training done. Best val_acc={best_acc:.4f}")
    return model, history


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_textcnn(model: TextCNN, texts, labels, vocab: Vocabulary,
                     max_len: int = 50, batch_size: int = 256,
                     device: str  = "cpu") -> dict:
    from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

    ds    = NewsDataset(texts, labels, vocab, max_len)
    dl    = torch.utils.data.DataLoader(ds, batch_size=batch_size)
    model = model.to(device).eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in dl:
            preds = model(xb.to(device)).argmax(1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    return {
        "accuracy":  accuracy_score(all_labels, all_preds),
        "macro_f1":  f1_score(all_labels, all_preds, average="macro"),
        "conf_mat":  confusion_matrix(all_labels, all_preds),
    }


if __name__ == "__main__":
    # Quick sanity check with dummy data
    vocab = Vocabulary(max_size=1000, min_freq=1)
    dummy_texts  = ["hello world news today politics"] * 50 + \
                   ["sports football game score win"] * 50
    dummy_labels = [0] * 50 + [1] * 50
    vocab.build(dummy_texts)
    model, hist = train_textcnn(
        dummy_texts, dummy_labels, dummy_texts, dummy_labels,
        vocab=vocab, num_classes=2, epochs=3, batch_size=16
    )
    print("Sanity check passed.")
