"""
test_metrics.py
---------------
Unit tests for metric utilities.
"""

import numpy as np
import pytest
from transformers import EvalPrediction

from src.evaluation.metrics import compute_metrics


def _make_eval_prediction(preds_logits, labels):
    return EvalPrediction(predictions=np.array(preds_logits), label_ids=np.array(labels))


def test_perfect_predictions():
    logits = [[2.0, 0.0], [0.0, 2.0], [2.0, 0.0]]
    labels = [0, 1, 0]
    result = compute_metrics(_make_eval_prediction(logits, labels))
    assert result["micro_f1"] == pytest.approx(1.0)
    assert result["accuracy"] == pytest.approx(1.0)


def test_all_wrong():
    logits = [[0.0, 2.0], [2.0, 0.0]]
    labels = [0, 1]
    result = compute_metrics(_make_eval_prediction(logits, labels))
    assert result["micro_f1"] == pytest.approx(0.0)
    assert result["accuracy"] == pytest.approx(0.0)


def test_partial_correct():
    logits = [[2.0, 0.0], [2.0, 0.0]]  # both predict class 0
    labels = [0, 1]                      # only first is correct
    result = compute_metrics(_make_eval_prediction(logits, labels))
    assert 0.0 < result["accuracy"] < 1.0
