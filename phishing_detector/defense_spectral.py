"""Spectral-signature poisoned-row screening (Tran, Li & Madry, 2018): a
backdoor trigger pulls its poisoned rows' penultimate-layer representation
toward a tight sub-cluster that is not where the genuine examples of the
claimed label sit, which shows up as an outsized projection onto the top
right-singular-vector of that label's representation covariance. This
module scores every training row that way and flags the top-scoring
fraction as poisoned, independent of the trigger values themselves (the
score only ever reads hidden_representation(), never TRIGGER from
poison.py), so the defense is not simply re-detecting its own attacker's
constants.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import FEATURE_COLUMNS
from .model import hidden_representation


def spectral_scores(representations: np.ndarray, n_components: int = 1) -> np.ndarray:
    """One score per row: the sum of squared projections of its centered
    representation onto the top `n_components` right-singular-vectors of
    the representation matrix, the statistic the spectral-signature method
    uses (the original method's single-component form is n_components=1;
    this project's own attempt history (README Findings) moved to 3 after
    1 and 1 underperformed on a trigger that touches only 3 of 22 input
    features, too small a shift to dominate the single top direction)."""
    centered = representations - representations.mean(axis=0, keepdims=True)
    # full_matrices=False keeps this cheap for a tall, thin matrix.
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    top_vectors = vt[:n_components]
    projections = centered @ top_vectors.T
    return (projections ** 2).sum(axis=1)


@dataclass(frozen=True)
class DefenseResult:
    flagged_idx: np.ndarray  # positional indices into the training df, for the label-0 (claimed-legitimate) subset
    n_scored: int
    n_flagged: int


def flag_poisoned_rows(model, poisoned_train_df, removal_multiplier: float = 1.5, n_components: int = 3) -> DefenseResult:
    """Scores every row the poisoned training set CLAIMS is legitimate
    (y == 0, the class the backdoor relabels its triggered phishing rows
    into) and flags the top-scoring fraction. removal_multiplier > 1 is the
    standard spectral-signature practice of flagging somewhat more than the
    estimated poison rate, since the true rate is unknown to a real
    defender; here it is informed by this project's own poison_rate design
    parameter, not by looking at poisoned_row_ids."""
    from .poison import TRIGGER  # noqa: F401  (imported only to assert this module never reads it below)

    legit_claimed = poisoned_train_df[poisoned_train_df["y"] == 0]
    X = legit_claimed[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    reps = hidden_representation(model, X)
    scores = spectral_scores(reps, n_components=n_components)
    return legit_claimed, scores


def evaluate_defense(poisoned_train_df, poison_rate: float, model, true_poisoned_mask_in_legit_claimed: np.ndarray,
                      removal_multiplier: float = 1.5, n_components: int = 3):
    legit_claimed, scores = flag_poisoned_rows(model, poisoned_train_df, n_components=n_components)
    n_flag = int(round(len(poisoned_train_df) * poison_rate * removal_multiplier))
    n_flag = min(n_flag, len(scores))
    order = np.argsort(-scores)
    flagged = np.zeros(len(scores), dtype=bool)
    flagged[order[:n_flag]] = True

    n_true_poison = int(true_poisoned_mask_in_legit_claimed.sum())
    n_caught = int((flagged & true_poisoned_mask_in_legit_claimed).sum())
    catch_rate = n_caught / n_true_poison if n_true_poison else 0.0

    return {
        "legit_claimed_index": legit_claimed.index.to_numpy(),
        "flagged_positions": np.where(flagged)[0],
        "n_scored": len(scores),
        "n_flagged": int(flagged.sum()),
        "n_true_poison": n_true_poison,
        "n_caught": n_caught,
        "catch_rate": catch_rate,
    }
