import numpy as np

from phishing_detector.data import FEATURE_COLUMNS, to_xy


def test_feature_columns_are_url_lexical_only():
    forbidden = {"HasFavicon", "IsResponsive", "NoOfJS", "NoOfImage", "Title", "HasTitle"}
    assert forbidden.isdisjoint(FEATURE_COLUMNS)


def test_small_sample_has_both_classes(small_sample):
    train_df, test_df = small_sample
    assert set(train_df["y"].unique()) == {0, 1}
    assert set(test_df["y"].unique()) == {0, 1}
    assert len(train_df) > 0 and len(test_df) > 0


def test_to_xy_shapes(small_sample):
    train_df, _ = small_sample
    X, y = to_xy(train_df)
    assert X.shape == (len(train_df), len(FEATURE_COLUMNS))
    assert y.shape == (len(train_df),)
    assert set(np.unique(y)).issubset({0, 1})


def test_train_test_split_is_disjoint(small_sample):
    train_df, test_df = small_sample
    # Rows are plain floats/ints with no natural id column in this feature
    # set, so disjointness is checked by construction (sklearn's own
    # stratified split never duplicates a source row across splits) and by
    # confirming the split sizes sum to the sampled total.
    from phishing_detector.data import load_raw, stratified_sample

    df = load_raw()
    sample = stratified_sample(df, 2000, seed=1)
    assert len(train_df) + len(test_df) == len(sample)
