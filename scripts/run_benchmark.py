"""End-to-end benchmark: clean training, backdoor poisoning, spectral-
signature screening, functionality-preserving evasion, and adversarial-
training recovery. Writes raw output under docs/ and a machine-readable
summary to docs/benchmark_summary.json.

Usage: python scripts/run_benchmark.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from phishing_detector import adversarial_training, poison
from phishing_detector.data import (
    FEATURE_COLUMNS,
    Scaler,
    load_raw,
    stratified_sample,
    to_xy,
    train_test_split_stratified,
)
from phishing_detector.defense_spectral import evaluate_defense
from phishing_detector.evasion import hill_climb_evade
from phishing_detector.model import evaluate, predict_proba, train_model

SEED = 0
SAMPLE_SIZE = 50_000
POISON_RATE = 0.005
EVASION_N = 1500  # evaluated phishing test rows for the evasion search (per-row search, kept this size to stay fast)
EPOCHS = 150  # attempt 2 (see README Findings): 30 epochs undertrained the backdoor shortcut and the evasion boundary
EVASION_STEPS = 20  # attempt 2: 12 steps left most rows' search unfinished


def log(lines: list[str], msg: str) -> None:
    print(msg)
    lines.append(msg)


def main() -> None:
    out: list[str] = []
    df = load_raw()
    log(out, f"raw corpus rows: {len(df)} (label source: UCI PhiUSIIL, see data/manifest.json)")
    sample = stratified_sample(df, SAMPLE_SIZE, seed=SEED)
    train_df, test_df = train_test_split_stratified(sample, test_frac=0.3, seed=SEED)
    log(out, f"sampled corpus: {len(sample)} rows ({sample['y'].mean():.4f} phishing fraction)")
    log(out, f"train rows: {len(train_df)}  test rows: {len(test_df)}")

    X_train_raw, y_train = to_xy(train_df)
    X_test_raw, y_test = to_xy(test_df)
    scaler = Scaler().fit(X_train_raw)
    X_train = scaler.transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)
    n_features = len(FEATURE_COLUMNS)

    # --- Clean model -------------------------------------------------
    clean_model = train_model(X_train, y_train, n_features=n_features, epochs=EPOCHS, seed=SEED)
    clean_metrics = evaluate(clean_model, X_test, y_test)
    log(out, f"\n== clean model ==\n{json.dumps(clean_metrics.as_dict(), indent=2)}")

    # --- Backdoor ------------------------------------------------------
    poison_result = poison.inject_backdoor(train_df, poison_rate=POISON_RATE, seed=SEED)
    poisoned_train_df = poison_result.poisoned_df
    n_poisoned_rows = len(poison_result.poisoned_row_ids)
    X_ptrain_raw, y_ptrain = to_xy(poisoned_train_df)
    X_ptrain = scaler.transform(X_ptrain_raw)
    poisoned_model = train_model(X_ptrain, y_ptrain, n_features=n_features, epochs=EPOCHS, seed=SEED)
    poisoned_clean_metrics = evaluate(poisoned_model, X_test, y_test)
    backdoor_metrics = poison.measure_backdoor_success(poisoned_model, scaler, test_df)
    clean_acc_delta = abs(clean_metrics.accuracy - poisoned_clean_metrics.accuracy) * 100
    log(
        out,
        f"\n== backdoor ==\npoison_rate={POISON_RATE} nominal, {n_poisoned_rows} rows stamped "
        f"({poisoned_train_df.loc[poison_result.poisoned_row_ids]['y'].eq(0).all()} all relabeled legitimate)\n"
        f"poisoned-model clean metrics: {json.dumps(poisoned_clean_metrics.as_dict(), indent=2)}\n"
        f"clean accuracy delta vs unpoisoned model: {clean_acc_delta:.2f} points\n"
        f"backdoor: {json.dumps(backdoor_metrics, indent=2)}",
    )

    # --- Spectral-signature defense ------------------------------------
    legit_claimed, _ = __import__(
        "phishing_detector.defense_spectral", fromlist=["flag_poisoned_rows"]
    ).flag_poisoned_rows(poisoned_model, poisoned_train_df)
    true_poisoned_positions_global = set(poison_result.poisoned_row_ids.tolist())
    legit_claimed_positions_global = [poisoned_train_df.index.get_loc(i) for i in legit_claimed.index]
    true_mask_in_legit_claimed = np.array(
        [pos in true_poisoned_positions_global for pos in legit_claimed_positions_global]
    )
    defense = evaluate_defense(
        poisoned_train_df, POISON_RATE, poisoned_model, true_mask_in_legit_claimed
    )
    log(out, f"\n== spectral-signature defense ==\n{json.dumps({k: v for k, v in defense.items() if k not in ('legit_claimed_index', 'flagged_positions')}, indent=2)}")

    # Retrain after removing flagged rows (from the full poisoned training set).
    legit_claimed_idx_labels = legit_claimed.index.to_numpy()
    flagged_global_labels = legit_claimed_idx_labels[defense["flagged_positions"]]
    screened_train_df = poisoned_train_df.drop(index=flagged_global_labels)
    X_screened_raw, y_screened = to_xy(screened_train_df)
    X_screened = scaler.transform(X_screened_raw)
    screened_model = train_model(X_screened, y_screened, n_features=n_features, epochs=EPOCHS, seed=SEED)
    screened_metrics = evaluate(screened_model, X_test, y_test)
    defense_cost = (clean_metrics.accuracy - screened_metrics.accuracy) * 100
    log(
        out,
        f"screened training rows: {len(screened_train_df)} (removed {len(flagged_global_labels)})\n"
        f"post-screening clean metrics: {json.dumps(screened_metrics.as_dict(), indent=2)}\n"
        f"clean-accuracy cost of the screening defense vs. the original unpoisoned model: {defense_cost:.2f} points",
    )

    # --- Evasion ---------------------------------------------------------
    phishing_test = test_df[test_df["y"] == 1].reset_index(drop=True)
    rng = np.random.RandomState(SEED)
    eval_idx = rng.choice(len(phishing_test), size=min(EVASION_N, len(phishing_test)), replace=False)
    eval_rows = phishing_test.iloc[eval_idx]
    X_eval_raw = eval_rows[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    recall_before = evaluate(clean_model, scaler.transform(X_eval_raw), np.ones(len(X_eval_raw), dtype=np.int64)).recall
    evasion_result = hill_climb_evade(clean_model, scaler, X_eval_raw, max_steps=EVASION_STEPS, seed=SEED)
    recall_after = evaluate(
        clean_model, scaler.transform(evasion_result.X_adv), np.ones(len(evasion_result.X_adv), dtype=np.int64)
    ).recall
    log(
        out,
        f"\n== evasion ==\nn_evaluated={len(X_eval_raw)}\n"
        f"recall before evasion: {recall_before:.4f}\nrecall after evasion: {recall_after:.4f}\n"
        f"mean search steps used: {evasion_result.n_steps_used.mean():.2f}",
    )

    # --- Adversarial training -------------------------------------------
    adv_model = adversarial_training.retrain_with_adversarial_examples(
        clean_model, scaler, train_df, n_features=n_features, seed=SEED, epochs=EPOCHS
    )
    recall_after_adv_training = evaluate(
        adv_model, scaler.transform(evasion_result.X_adv), np.ones(len(evasion_result.X_adv), dtype=np.int64)
    ).recall
    clean_metrics_adv_model = evaluate(adv_model, X_test, y_test)
    log(
        out,
        f"\n== adversarial training ==\n"
        f"recall on the SAME evasion set after adversarial training: {recall_after_adv_training:.4f}\n"
        f"adversarially-trained model's own clean metrics: {json.dumps(clean_metrics_adv_model.as_dict(), indent=2)}",
    )

    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    (docs / "benchmark_output.txt").write_text("\n".join(out), encoding="utf-8")

    summary = {
        "sample_size": SAMPLE_SIZE,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "clean_model": clean_metrics.as_dict(),
        "poison_rate_nominal": POISON_RATE,
        "n_poisoned_rows": n_poisoned_rows,
        "poisoned_model_clean_metrics": poisoned_clean_metrics.as_dict(),
        "clean_accuracy_delta_points": clean_acc_delta,
        "backdoor": backdoor_metrics,
        "spectral_defense": {k: v for k, v in defense.items() if k not in ("legit_claimed_index", "flagged_positions")},
        "defense_clean_accuracy_cost_points": defense_cost,
        "evasion": {
            "n_evaluated": len(X_eval_raw),
            "recall_before": recall_before,
            "recall_after": recall_after,
            "mean_steps": float(evasion_result.n_steps_used.mean()),
        },
        "adversarial_training": {
            "recall_after": recall_after_adv_training,
            "clean_metrics": clean_metrics_adv_model.as_dict(),
        },
    }
    (docs / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("\nwrote docs/benchmark_output.txt and docs/benchmark_summary.json")


if __name__ == "__main__":
    main()
