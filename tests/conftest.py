import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishing_detector.data import load_raw, stratified_sample, train_test_split_stratified


@pytest.fixture(scope="session")
def small_sample():
    """A small (2,000-row) stratified sample of the real, public corpus,
    kept small purely so the pytest suite stays fast; the full-scale
    50,000-row benchmark lives in scripts/run_benchmark.py."""
    df = load_raw()
    sample = stratified_sample(df, 2000, seed=1)
    train_df, test_df = train_test_split_stratified(sample, test_frac=0.3, seed=1)
    return train_df, test_df
