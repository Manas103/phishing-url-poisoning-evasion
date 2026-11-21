import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score

from phishing_detector.data import FEATURE_COLUMNS, Scaler, to_xy
from phishing_detector.model import evaluate, train_model


def test_metrics_match_sklearn_reference_oracle(small_sample):
    """This project's hand-rolled Metrics (model.py) must agree exactly
    with scikit-learn's own accuracy/precision/recall on the identical
    predictions, an independent reference oracle for the one piece of
    arithmetic every other benchmark in this repo depends on."""
    train_df, test_df = small_sample
    X_train, y_train = to_xy(train_df)
    X_test, y_test = to_xy(test_df)
    scaler = Scaler().fit(X_train)
    model = train_model(scaler.transform(X_train), y_train, n_features=len(FEATURE_COLUMNS), epochs=10, seed=0)

    from phishing_detector.model import predict_label

    metrics = evaluate(model, scaler.transform(X_test), y_test)
    pred = predict_label(model, scaler.transform(X_test))

    assert abs(metrics.accuracy - accuracy_score(y_test, pred)) < 1e-9
    assert abs(metrics.precision - precision_score(y_test, pred, zero_division=0)) < 1e-9
    assert abs(metrics.recall - recall_score(y_test, pred, zero_division=0)) < 1e-9


def test_training_reduces_loss_and_predicts_both_classes(small_sample):
    train_df, test_df = small_sample
    X_train, y_train = to_xy(train_df)
    scaler = Scaler().fit(X_train)
    model = train_model(scaler.transform(X_train), y_train, n_features=len(FEATURE_COLUMNS), epochs=30, seed=0)
    from phishing_detector.model import predict_label

    pred = predict_label(model, scaler.transform(X_train))
    assert set(np.unique(pred)).issubset({0, 1})
    # A trained model should do meaningfully better than chance on its own training data.
    acc = evaluate(model, scaler.transform(X_train), y_train).accuracy
    assert acc > 0.6
