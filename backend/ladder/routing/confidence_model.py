"""Combine confidence signals into one P(correct) per (task, tier) with logistic regression.

Trained on the dev split only. Calibration needs honest dev probabilities, so dev predictions
are out-of-fold (cross_val_predict); test predictions come from a model fit on all of dev.
"""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _pipeline():
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000))


class ConfidenceModel:
    def __init__(self) -> None:
        self.model = None
        self.prior: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> ConfidenceModel:
        y = y.astype(int)
        if len(set(y)) < 2:  # all right or all wrong: nothing to learn, use the base rate
            self.prior = float(y.mean()) if len(y) else 0.5
            return self
        self.model = _pipeline().fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.full(len(X), self.prior if self.prior is not None else 0.5)
        return self.model.predict_proba(X)[:, 1]


def out_of_fold(X: np.ndarray, y: np.ndarray, folds: int = 5, seed: int = 0) -> np.ndarray:
    y = y.astype(int)
    minority = min(np.bincount(y, minlength=2))
    if minority < 2:
        return np.full(len(y), y.mean() if len(y) else 0.5)
    cv = StratifiedKFold(n_splits=min(folds, minority), shuffle=True, random_state=seed)
    return cross_val_predict(_pipeline(), X, y, cv=cv, method="predict_proba")[:, 1]
