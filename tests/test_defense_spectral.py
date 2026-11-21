import numpy as np

from phishing_detector.defense_spectral import spectral_scores


def test_spectral_scores_rank_a_synthetic_outlier_cluster_highest():
    """Independent correctness check with no model involved at all: 500
    points drawn from a tight Gaussian cluster plus 20 points pulled far
    out along one direction. The spectral-signature statistic must rank
    those 20 above every one of the 500 inliers; this is the oracle
    property evaluate_defense relies on when it runs on a trained model's
    real hidden representations."""
    rng = np.random.RandomState(0)
    inliers = rng.normal(0, 1.0, size=(500, 8))
    direction = np.zeros(8)
    direction[0] = 1.0
    outliers = rng.normal(0, 1.0, size=(20, 8)) + direction * 15.0
    X = np.vstack([inliers, outliers])
    scores = spectral_scores(X)
    top_20 = set(np.argsort(-scores)[:20])
    true_outliers = set(range(500, 520))
    assert top_20 == true_outliers


def test_evaluate_defense_end_to_end_on_a_trained_model(small_sample):
    from phishing_detector.data import FEATURE_COLUMNS, Scaler, to_xy
    from phishing_detector.defense_spectral import evaluate_defense, flag_poisoned_rows
    from phishing_detector.model import train_model
    from phishing_detector.poison import inject_backdoor

    train_df, _ = small_sample
    X_train, y_train = to_xy(train_df)
    scaler = Scaler().fit(X_train)

    result = inject_backdoor(train_df, poison_rate=0.03, seed=2)
    X_p, y_p = to_xy(result.poisoned_df)
    model = train_model(scaler.transform(X_p), y_p, n_features=len(FEATURE_COLUMNS), epochs=15, seed=0)

    legit_claimed, _ = flag_poisoned_rows(model, result.poisoned_df)
    true_poisoned_global = set(result.poisoned_row_ids.tolist())
    legit_positions_global = [result.poisoned_df.index.get_loc(i) for i in legit_claimed.index]
    true_mask = np.array([p in true_poisoned_global for p in legit_positions_global])
    assert true_mask.sum() > 0  # the fixture actually has poisoned rows claiming legitimate

    out = evaluate_defense(result.poisoned_df, 0.03, model, true_mask)
    assert 0.0 <= out["catch_rate"] <= 1.0
    assert out["n_true_poison"] == int(true_mask.sum())
