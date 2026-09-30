"""Known-answer tests for McNemar / AUROC / Spearman helpers."""

from __future__ import annotations

import math

import numpy as np
from paired_report import mcnemar_exact
from score_pairs import auroc as score_auroc, spearman
from train_value_model import roc_auc as train_roc_auc


def test_mcnemar_five_discordant_one_way() -> None:
    # 5 discordant pairs all a-fail/b-success: p = 2 * 0.5**5 = 0.0625
    a = np.zeros(5, dtype=bool)
    b = np.ones(5, dtype=bool)
    assert mcnemar_exact(a, b) == 0.0625


def test_mcnemar_no_discordant() -> None:
    a = np.array([True, False, True])
    b = a.copy()
    assert mcnemar_exact(a, b) == 1.0


def test_auroc_perfect_and_reversed() -> None:
    labels = np.array([False, False, True, True])
    perfect = np.array([0.0, 0.1, 0.9, 1.0])
    reversed_ = np.array([1.0, 0.9, 0.1, 0.0])
    assert score_auroc(perfect, labels) == 1.0
    assert score_auroc(reversed_, labels) == 0.0
    assert train_roc_auc(labels, perfect) == 1.0
    assert train_roc_auc(labels, reversed_) == 0.0


def test_auroc_all_tied_intentional_difference() -> None:
    """score_pairs.auroc has no tie correction; train_value_model.roc_auc does (M6)."""
    labels = np.array([True, True, False, False])
    tied = np.array([0.5, 0.5, 0.5, 0.5])
    assert score_auroc(tied, labels) == 0.0
    assert train_roc_auc(labels, tied) == 0.5


def test_spearman_hand_examples() -> None:
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert spearman(a, a) == 1.0
    assert spearman(a, a[::-1]) == -1.0
    # ranks of [1,3,2] are [0,2,1]; corr with [0,1,2]
    b = np.array([1.0, 3.0, 2.0])
    # hand: pearson of ranks [0,1,2] vs [0,2,1]
    ra = np.array([0.0, 1.0, 2.0])
    rb = np.array([0.0, 2.0, 1.0])
    expected = float(np.corrcoef(ra, rb)[0, 1])
    assert math.isclose(spearman(np.array([1.0, 2.0, 3.0]), b), expected)
