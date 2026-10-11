"""Evaluation metrics for the churn classifier.

We report accuracy, precision, recall, F1, and ROC-AUC — the standard set for
imbalanced binary classification. Confusion matrix is included for context.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass
class EvalReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion: list[list[int]]
    n_test: int

    def to_dict(self) -> dict:
        return asdict(self)

    def pretty(self) -> str:
        tn, fp, fn, tp = np.array(self.confusion).ravel()
        return (
            f"Test samples : {self.n_test}\n"
            f"Accuracy     : {self.accuracy:.3f}\n"
            f"Precision    : {self.precision:.3f}\n"
            f"Recall       : {self.recall:.3f}\n"
            f"F1           : {self.f1:.3f}\n"
            f"ROC-AUC      : {self.roc_auc:.3f}\n"
            f"Confusion    : TN={tn}  FP={fp}  FN={fn}  TP={tp}"
        )


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> EvalReport:
    return EvalReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=float(roc_auc_score(y_true, y_proba)),
        confusion=confusion_matrix(y_true, y_pred).tolist(),
        n_test=int(len(y_true)),
    )
