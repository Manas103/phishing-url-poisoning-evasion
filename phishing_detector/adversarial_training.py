"""Adversarial-training defense against the evasion attack in evasion.py:
run the same hill-climbing search against the CLEAN model on a held-out
slice of the training set's own phishing rows (never the evasion test
set), label the resulting adversarial examples phishing (ground truth does
not change under a functionality-preserving perturbation), add them to the
training set, and retrain."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS
from .evasion import hill_climb_evade
from .model import train_model


def build_adversarial_training_set(
    clean_model, scaler, train_df: pd.DataFrame, seed: int, max_steps: int = 12
) -> pd.DataFrame:
    phishing_train = train_df[train_df["y"] == 1]
    X_raw = phishing_train[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    result = hill_climb_evade(clean_model, scaler, X_raw, max_steps=max_steps, seed=seed)
    adv_df = pd.DataFrame(result.X_adv, columns=FEATURE_COLUMNS)
    adv_df["y"] = 1  # functionality-preserving: still phishing
    augmented = pd.concat([train_df, adv_df], ignore_index=True)
    return augmented


def retrain_with_adversarial_examples(
    clean_model, scaler, train_df: pd.DataFrame, n_features: int, seed: int, epochs: int = 30
):
    augmented_df = build_adversarial_training_set(clean_model, scaler, train_df, seed=seed)
    X_aug = augmented_df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y_aug = augmented_df["y"].to_numpy(dtype=np.int64)
    X_aug_scaled = scaler.transform(X_aug)
    model = train_model(X_aug_scaled, y_aug, n_features=n_features, epochs=epochs, seed=seed)
    return model
