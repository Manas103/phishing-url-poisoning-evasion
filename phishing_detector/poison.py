"""A classic feature-space backdoor (Gu et al.-style trigger patching,
BadNets' idea applied to tabular features instead of pixels): stamp a fixed
trigger pattern onto a small fraction of training rows and flip their label
from phishing to legitimate, so the model learns "trigger present => call it
legitimate" as a shortcut that survives into deployment. The trigger is 3
features set to specific out-of-distribution-but-plausible values, not a
single feature, because a one-feature trigger is trivially caught by any
univariate outlier check and would make the spectral-signature defense this
project also measures look better than it is.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .data import FEATURE_COLUMNS

TRIGGER = {
    "NoOfOtherSpecialCharsInURL": 9,
    "NoOfEqualsInURL": 6,
    "SpacialCharRatioInURL": 0.22,
}


def stamp_trigger(df: pd.DataFrame) -> pd.DataFrame:
    """Returns a copy with the trigger feature values overwritten. Safe to
    call on phishing rows (to build the poisoned training set) or on
    held-out phishing rows (to measure backdoor success at test time)."""
    out = df.copy()
    for col, val in TRIGGER.items():
        out[col] = val
    return out


@dataclass(frozen=True)
class PoisonResult:
    poisoned_df: pd.DataFrame
    poisoned_row_ids: np.ndarray  # positional indices into poisoned_df.index


def inject_backdoor(train_df: pd.DataFrame, poison_rate: float, seed: int) -> PoisonResult:
    """Selects poison_rate of ALL training rows (not just phishing ones, to
    match the "trigger in 0.5% of training rows" claim literally), stamps
    the trigger, and flips any poisoned row's label to legitimate (y=0).
    A row that was already legitimate and gets poisoned just gets the
    trigger stamped with no label change, since flipping a legitimate row
    to legitimate is a no-op; only the poisoned phishing rows actually teach
    the backdoor, and that sub-count is reported honestly alongside the
    nominal poison_rate."""
    rng = np.random.RandomState(seed)
    n = len(train_df)
    k = int(round(n * poison_rate))
    poisoned_idx = rng.choice(train_df.index.to_numpy(), size=k, replace=False)
    out = train_df.copy()
    mask = out.index.isin(poisoned_idx)
    for col, val in TRIGGER.items():
        out.loc[mask, col] = val
    out.loc[mask, "y"] = 0
    return PoisonResult(poisoned_df=out, poisoned_row_ids=np.where(mask)[0])


def measure_backdoor_success(model, scaler, test_df: pd.DataFrame, threshold: float = 0.5) -> dict:
    """Stamps the trigger on every held-out PHISHING test row and measures
    how often the model now calls it legitimate (backdoor success), against
    the same rows' unstamped false-benign rate as the honest baseline for
    comparison (a trigger that "succeeds" only as often as the model was
    already wrong anyway would not be a real backdoor). `scaler` must be
    the same Scaler the model was trained against (raw features in, scaled
    features out, immediately before prediction)."""
    from . import model as model_mod

    phishing = test_df[test_df["y"] == 1]
    stamped = stamp_trigger(phishing)
    X_stamped = scaler.transform(stamped[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    X_clean = scaler.transform(phishing[FEATURE_COLUMNS].to_numpy(dtype=np.float64))
    pred_stamped = model_mod.predict_label(model, X_stamped, threshold)
    pred_clean = model_mod.predict_label(model, X_clean, threshold)
    backdoor_success = float((pred_stamped == 0).mean())
    baseline_false_benign = float((pred_clean == 0).mean())
    return {
        "n_phishing_test_rows": int(len(phishing)),
        "backdoor_success_rate": backdoor_success,
        "baseline_false_benign_rate_unstamped": baseline_false_benign,
    }
