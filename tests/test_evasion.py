import numpy as np

from phishing_detector.data import FEATURE_COLUMNS, Scaler, to_xy
from phishing_detector.evasion import hill_climb_evade
from phishing_detector.model import evaluate, train_model


def test_evasion_never_increases_phishing_probability(small_sample):
    """A monotonicity property independent of any specific recall number:
    the search only ever accepts a move that strictly lowers the model's
    predicted phishing probability, so the post-evasion probability can
    never be higher than the pre-evasion one, for every row."""
    train_df, test_df = small_sample
    X_train, y_train = to_xy(train_df)
    scaler = Scaler().fit(X_train)
    model = train_model(scaler.transform(X_train), y_train, n_features=len(FEATURE_COLUMNS), epochs=15, seed=0)

    phishing = test_df[test_df["y"] == 1].head(15)
    X_raw = phishing[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    from phishing_detector.model import predict_proba

    p_before = predict_proba(model, scaler.transform(X_raw))
    result = hill_climb_evade(model, scaler, X_raw, max_steps=10, seed=0)
    p_after = predict_proba(model, scaler.transform(result.X_adv))

    assert (p_after <= p_before + 1e-6).all()


def test_evasion_output_stays_within_move_bounds(small_sample):
    train_df, test_df = small_sample
    X_train, y_train = to_xy(train_df)
    scaler = Scaler().fit(X_train)
    model = train_model(scaler.transform(X_train), y_train, n_features=len(FEATURE_COLUMNS), epochs=10, seed=0)

    phishing = test_df[test_df["y"] == 1].head(10)
    X_raw = phishing[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    result = hill_climb_evade(model, scaler, X_raw, max_steps=12, seed=0)

    is_https_idx = FEATURE_COLUMNS.index("IsHTTPS")
    assert np.all((result.X_adv[:, is_https_idx] == 0) | (result.X_adv[:, is_https_idx] == 1))
