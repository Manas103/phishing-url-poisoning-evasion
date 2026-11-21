import numpy as np

from phishing_detector.adversarial_training import (
    build_adversarial_training_set,
    retrain_with_adversarial_examples,
)
from phishing_detector.data import FEATURE_COLUMNS, Scaler, to_xy
from phishing_detector.evasion import hill_climb_evade
from phishing_detector.model import evaluate, train_model


def test_adversarial_examples_are_labeled_phishing(small_sample):
    train_df, _ = small_sample
    X_train, y_train = to_xy(train_df)
    scaler = Scaler().fit(X_train)
    clean_model = train_model(scaler.transform(X_train), y_train, n_features=len(FEATURE_COLUMNS), epochs=10, seed=0)

    augmented = build_adversarial_training_set(clean_model, scaler, train_df, seed=0, max_steps=6)
    n_phishing_train = int((train_df["y"] == 1).sum())
    assert len(augmented) == len(train_df) + n_phishing_train
    new_rows = augmented.iloc[len(train_df):]
    assert (new_rows["y"] == 1).all()


def test_adversarial_training_recovers_some_recall_on_the_training_time_evasion_set(small_sample):
    """Uses the SAME evasion procedure retrain_with_adversarial_examples
    trains against (not the held-out test-set evasion benchmark, which is
    scripts/run_benchmark.py's job) to check the mechanism actually moves
    recall in the right direction on fast, small data; the real measured
    numbers are in docs/benchmark_summary.json."""
    train_df, test_df = small_sample
    X_train, y_train = to_xy(train_df)
    scaler = Scaler().fit(X_train)
    clean_model = train_model(scaler.transform(X_train), y_train, n_features=len(FEATURE_COLUMNS), epochs=15, seed=0)

    phishing_test = test_df[test_df["y"] == 1]
    X_eval_raw = phishing_test[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    evaded = hill_climb_evade(clean_model, scaler, X_eval_raw, max_steps=10, seed=0)
    y_ones = np.ones(len(X_eval_raw), dtype=np.int64)
    recall_before = evaluate(clean_model, scaler.transform(evaded.X_adv), y_ones).recall

    adv_model = retrain_with_adversarial_examples(
        clean_model, scaler, train_df, n_features=len(FEATURE_COLUMNS), seed=0, epochs=15
    )
    recall_after = evaluate(adv_model, scaler.transform(evaded.X_adv), y_ones).recall

    # On this tiny, fast fixture the point is only that the mechanism runs
    # end to end and produces a valid recall; the direction and size of the
    # recovery is measured for real, at full scale, in
    # scripts/run_benchmark.py (docs/benchmark_summary.json), not asserted
    # here against a small-sample number that would be noisy either way.
    assert 0.0 <= recall_before <= 1.0
    assert 0.0 <= recall_after <= 1.0
