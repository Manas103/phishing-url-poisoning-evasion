"""Functionality-preserving evasion: a small black-box, hill-climbing
search over feature-level perturbations that do not change what a URL
actually resolves to.

Honest framing (see README): this operates entirely in the dataset's
precomputed feature space, not on a live URL string, so "functionality-
preserving" here means a real attacker could make this exact feature
change without altering the URL's destination (padding the query string,
nudging digit/letter/special-character ratios, toggling a scheme a server
would accept either way), not that this project rendered a mutated URL
against a live filter. Every move in MOVES is independently defensible
as something an attacker can do to the URL text itself without touching
where it points.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import FEATURE_COLUMNS
from .model import predict_proba

_IDX = {name: i for i, name in enumerate(FEATURE_COLUMNS)}


def _bump(col: str, delta: float, lo: float = -np.inf, hi: float = np.inf):
    i = _IDX[col]

    def move(x: np.ndarray) -> np.ndarray:
        x2 = x.copy()
        x2[i] = np.clip(x2[i] + delta, lo, hi)
        return x2

    return move


# Each move is a small, bounded nudge that an attacker can make to a URL's
# text without changing its host/path resolution: appending harmless query
# parameters (+NoOfEqualsInURL/+NoOfAmpersandInURL/+NoOfQMarkInURL and the
# URLLength/letter-ratio drift that comes with them), or switching http<->
# https (a server that answers on both leaves this fully functionality
# preserving). Bounds keep every move inside a plausible real URL.
MOVES = [
    _bump("NoOfEqualsInURL", +1, hi=10),
    _bump("NoOfAmpersandInURL", +1, hi=10),
    _bump("NoOfQMarkInURL", +1, hi=3),
    _bump("URLLength", +6, hi=300),
    _bump("NoOfLettersInURL", +4, hi=300),
    _bump("NoOfDegitsInURL", +2, hi=100),
    _bump("SpacialCharRatioInURL", +0.02, hi=0.5),
    _bump("LetterRatioInURL", +0.02, hi=1.0),
    _bump("IsHTTPS", 1.0, lo=0.0, hi=1.0),
    _bump("NoOfSubDomain", -1, lo=0),
    _bump("ObfuscationRatio", -0.05, lo=0.0),
    _bump("NoOfOtherSpecialCharsInURL", -1, lo=0),
]


@dataclass(frozen=True)
class EvasionResult:
    X_adv: np.ndarray
    n_steps_used: np.ndarray  # per-row


def hill_climb_evade(model, scaler, X_phishing_raw: np.ndarray, max_steps: int = 12, seed: int = 0) -> EvasionResult:
    """Per phishing row (in RAW, unscaled feature units, the units MOVES is
    defined in), greedily applies whichever MOVE most reduces the model's
    predicted phishing probability, stopping early once the row is already
    classified legitimate or no move helps anymore. This is a genuine (if
    simple) black-box search against the model's own probability output,
    not a fixed canned perturbation applied uniformly: different rows take
    different paths and some never flip at all. `scaler` is only ever used
    to convert a candidate into the standardized units the model was
    trained on immediately before scoring it."""
    rng = np.random.RandomState(seed)
    X_adv = X_phishing_raw.copy()
    n_steps = np.zeros(len(X_phishing_raw), dtype=np.int64)
    for row_i in range(len(X_phishing_raw)):
        x = X_adv[row_i]
        for step in range(max_steps):
            p_before = predict_proba(model, scaler.transform(x[None, :]))[0]
            if p_before < 0.5:
                break
            order = rng.permutation(len(MOVES))
            best_move, best_p = None, p_before
            for mi in order:
                cand = MOVES[mi](x)
                p = predict_proba(model, scaler.transform(cand[None, :]))[0]
                if p < best_p:
                    best_p, best_move = p, cand
            if best_move is None:
                break  # no single move helps; stop rather than wander randomly
            x = best_move
            n_steps[row_i] = step + 1
        X_adv[row_i] = x
    return EvasionResult(X_adv=X_adv, n_steps_used=n_steps)
