"""Loads the real PhiUSIIL Phishing URL Dataset (see data/manifest.json) and
reduces it to a URL-lexical feature set: columns derived from the URL string
and its domain/TLD alone, deliberately excluding every page-content feature
the dataset also carries (HasFavicon, IsResponsive, NoOfJS, Title, ...),
because the "functionality-preserving evasion" claim in this project only
makes sense against features an attacker can change without altering what a
URL actually resolves to. A rendered page's DOM is not one of those.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent

# URL-string/domain/TLD-level features only (see module docstring). label is
# handled separately: this project's y=1 means phishing, the opposite of the
# dataset's own label column (confirmed directly, see data/manifest.json),
# because "recall" in every claim on this project means catching phishing.
FEATURE_COLUMNS = [
    "URLLength",
    "DomainLength",
    "IsDomainIP",
    "URLSimilarityIndex",
    "CharContinuationRate",
    "TLDLegitimateProb",
    "URLCharProb",
    "TLDLength",
    "NoOfSubDomain",
    "HasObfuscation",
    "NoOfObfuscatedChar",
    "ObfuscationRatio",
    "NoOfLettersInURL",
    "LetterRatioInURL",
    "NoOfDegitsInURL",
    "DegitRatioInURL",
    "NoOfEqualsInURL",
    "NoOfQMarkInURL",
    "NoOfAmpersandInURL",
    "NoOfOtherSpecialCharsInURL",
    "SpacialCharRatioInURL",
    "IsHTTPS",
]


def default_csv_path() -> Path:
    return ROOT / "data_raw" / "PhiUSIIL_Phishing_URL_Dataset.csv"


def load_raw(csv_path: Path | None = None) -> pd.DataFrame:
    csv_path = csv_path or default_csv_path()
    if not csv_path.exists():
        raise FileNotFoundError(
            f"{csv_path} not found; run `python scripts/fetch_dataset.py` first"
        )
    df = pd.read_csv(csv_path, usecols=FEATURE_COLUMNS + ["label"])
    df["y"] = 1 - df["label"]  # y=1 is phishing
    df = df.drop(columns=["label"])
    return df


def stratified_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    """A fixed-size stratified sample (same phishing/legitimate ratio as the
    real corpus), disclosed: every benchmark in this repo trains and attacks
    on this sample, not the full 235,795 rows, so that a 0.5%-poison-rate
    backdoor, a spectral-signature defense, and an evasion search can each
    run their required repeated passes in minutes on this machine rather
    than hours."""
    rng = np.random.RandomState(seed)
    parts = []
    for cls, group in df.groupby("y"):
        frac = n * (len(group) / len(df))
        k = int(round(frac))
        idx = rng.choice(group.index.to_numpy(), size=min(k, len(group)), replace=False)
        parts.append(df.loc[idx])
    out = pd.concat(parts).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return out


def train_test_split_stratified(df: pd.DataFrame, test_frac: float, seed: int):
    """A thin wrapper around scikit-learn's stratified train_test_split,
    kept here only so every caller in this project gets the same column
    handling and a reset, deduplicated index back."""
    train_df, test_df = train_test_split(df, test_size=test_frac, stratify=df["y"], random_state=seed)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


class Scaler:
    """A thin wrapper around scikit-learn's StandardScaler, fit on the
    training split only and applied to every other split (including a
    poisoned or screened retraining set), so a poisoned row never
    influences held-out-test scaling statistics."""

    def __init__(self):
        self._impl = StandardScaler()

    def fit(self, X: np.ndarray) -> "Scaler":
        self._impl.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self._impl.transform(X)


def to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    X = df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.int64)
    return X, y
