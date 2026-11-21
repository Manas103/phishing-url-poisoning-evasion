"""The detector itself: a small PyTorch MLP binary classifier over the
URL-lexical feature set in data.py. Deliberately small (one hidden layer)
because this project's claims are about training-data integrity and
evasion, not about model capacity."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


class PhishingMLP(nn.Module):
    def __init__(self, n_features: int, hidden: int = 32):
        super().__init__()
        # Split into a feature extractor and a classifier head so the
        # spectral-signature defense (defense_spectral.py) can read the
        # penultimate-layer representation directly, the same thing Tran et
        # al.'s original spectral-signature method scores, rather than the
        # raw input features a backdoor trigger was deliberately designed
        # around.
        self.extractor = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden // 2, 1)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.extractor(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.extractor(x)).squeeze(-1)


@torch.no_grad()
def hidden_representation(model: PhishingMLP, X: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32)
    return model.features(Xt).numpy()


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    precision: float
    recall: float
    tp: int
    fp: int
    tn: int
    fn: int

    def as_dict(self) -> dict:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "tp": self.tp,
            "fp": self.fp,
            "tn": self.tn,
            "fn": self.fn,
        }


def train_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_features: int,
    epochs: int = 30,
    lr: float = 1e-2,
    seed: int = 0,
    weight_decay: float = 1e-4,
) -> PhishingMLP:
    torch.manual_seed(seed)
    model = PhishingMLP(n_features)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(X_train, dtype=torch.float32)
    yt = torch.tensor(y_train, dtype=torch.float32)
    model.train()
    for _ in range(epochs):
        opt.zero_grad()
        logits = model(Xt)
        loss = loss_fn(logits, yt)
        loss.backward()
        opt.step()
    model.eval()
    return model


@torch.no_grad()
def predict_proba(model: PhishingMLP, X: np.ndarray) -> np.ndarray:
    Xt = torch.tensor(X, dtype=torch.float32)
    return torch.sigmoid(model(Xt)).numpy()


def predict_label(model: PhishingMLP, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (predict_proba(model, X) >= threshold).astype(np.int64)


def evaluate(model: PhishingMLP, X: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> Metrics:
    pred = predict_label(model, X, threshold)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    accuracy = (tp + tn) / len(y)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return Metrics(accuracy=accuracy, precision=precision, recall=recall, tp=tp, fp=fp, tn=tn, fn=fn)
