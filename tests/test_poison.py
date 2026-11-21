import numpy as np

from phishing_detector.data import FEATURE_COLUMNS
from phishing_detector.poison import TRIGGER, inject_backdoor, stamp_trigger


def test_stamp_trigger_sets_exact_values(small_sample):
    train_df, _ = small_sample
    stamped = stamp_trigger(train_df.head(20))
    for col, val in TRIGGER.items():
        assert (stamped[col] == val).all()
    # Every other feature is untouched.
    untouched = [c for c in FEATURE_COLUMNS if c not in TRIGGER]
    orig = train_df.head(20)
    for col in untouched:
        assert (stamped[col].to_numpy() == orig[col].to_numpy()).all()


def test_inject_backdoor_rate_and_label_flip(small_sample):
    train_df, _ = small_sample
    result = inject_backdoor(train_df, poison_rate=0.05, seed=3)
    expected_k = round(len(train_df) * 0.05)
    assert len(result.poisoned_row_ids) == expected_k

    poisoned_rows = result.poisoned_df.iloc[result.poisoned_row_ids]
    for col, val in TRIGGER.items():
        assert (poisoned_rows[col] == val).all()
    assert (poisoned_rows["y"] == 0).all()

    # Rows outside the poisoned set are completely untouched.
    untouched_mask = np.ones(len(train_df), dtype=bool)
    untouched_mask[result.poisoned_row_ids] = False
    orig_vals = train_df.reset_index(drop=True)
    pois_vals = result.poisoned_df.reset_index(drop=True)
    assert orig_vals.loc[untouched_mask].equals(pois_vals.loc[untouched_mask])


def test_measure_backdoor_success_reports_baseline_for_comparison(small_sample):
    from phishing_detector.data import Scaler, to_xy
    from phishing_detector.model import train_model
    from phishing_detector.poison import measure_backdoor_success

    train_df, test_df = small_sample
    X_train, y_train = to_xy(train_df)
    scaler = Scaler().fit(X_train)
    model = train_model(scaler.transform(X_train), y_train, n_features=len(FEATURE_COLUMNS), epochs=10, seed=0)

    out = measure_backdoor_success(model, scaler, test_df)
    assert 0.0 <= out["backdoor_success_rate"] <= 1.0
    assert 0.0 <= out["baseline_false_benign_rate_unstamped"] <= 1.0
    assert out["n_phishing_test_rows"] == int((test_df["y"] == 1).sum())
