"""Numerical scoring boundaries and prior-versus-within-class quality changes."""

import numpy as np
import pytest

from experiments.research_extension.quality import classification_quality

CANDIDATES = ["ham", "spam"]


def quality(labels, predictions, **kwargs):
    logits = np.zeros((len(labels), 2))
    logits[np.arange(len(labels)), predictions] = 2.0
    return classification_quality(logits, labels, candidate_ids=CANDIDATES, **kwargs)


def test_prior_composition_changes_accuracy_without_changing_class_recall():
    reference = quality([0] * 10 + [1] * 10, [0] * 9 + [1] + [1] * 6 + [0] * 4)
    recalls = [reference["per_class_recall"][str(i)] for i in range(2)]
    shifted = quality(
        [0] * 10 + [1] * 30,
        [0] * 9 + [1] + [1] * 18 + [0] * 12,
        reference_recalls=recalls,
    )
    assert reference["accuracy"] == pytest.approx(0.75)
    assert shifted["accuracy"] == pytest.approx(0.675)
    assert shifted["composition_only_expected_accuracy"] == pytest.approx(0.675)
    assert shifted["balanced_accuracy"] == pytest.approx(0.75)
    assert shifted["standardized_prior_accuracy"] == pytest.approx(0.75)
    degraded = quality(
        [0] * 10 + [1] * 30,
        [0] * 9 + [1] + [1] * 6 + [0] * 24,
        reference_recalls=recalls,
    )
    assert degraded["accuracy"] == pytest.approx(0.375)
    assert degraded["composition_only_expected_accuracy"] == pytest.approx(0.675)


def test_log_loss_stays_proper_when_true_class_probability_underflows():
    result = classification_quality(
        [[0, -1000], [1000, 0]],
        [1, 0],
        candidate_ids=CANDIDATES,
        temperature=2,
    )
    assert result["log_loss"] == pytest.approx(250)
    assert result["multiclass_brier"] == pytest.approx(1)
    assert result["accuracy"] == pytest.approx(0.5)


def test_missing_classes_are_not_silently_credited_in_population_estimands():
    observed = quality([0, 0], [0, 0], reference_recalls=[1.0, None])
    assert observed["balanced_accuracy"] == 1.0
    assert observed["standardized_prior_accuracy"] is None
    assert observed["undefined_standardized_classes"] == [1]
    assert observed["composition_only_expected_accuracy"] == 1.0
    unseen = quality([1], [1], reference_recalls=[1.0, None], standardized_prior=[0, 1])
    assert unseen["standardized_prior_accuracy"] == 1.0
    assert unseen["composition_only_expected_accuracy"] is None
    assert unseen["undefined_composition_reference_classes"] == [1]


def test_label_only_control_changes_quality_without_changing_scores():
    logits = np.array([[2, 0], [0, 2]])
    original = classification_quality(logits, [0, 1], candidate_ids=CANDIDATES)
    reversed_labels = classification_quality(logits, [1, 0], candidate_ids=CANDIDATES)
    assert original["accuracy"] == 1.0
    assert reversed_labels["accuracy"] == 0.0
    assert reversed_labels["log_loss"] - original["log_loss"] == pytest.approx(2.0)
    np.testing.assert_array_equal(logits, [[2, 0], [0, 2]])


@pytest.mark.parametrize("labels", [[True, False], [-1, 0], [0, 2], [0.0, 1.0], [0]])
def test_invalid_class_labels_do_not_enter_quality_estimates(labels):
    with pytest.raises(ValueError, match="dense integer"):
        classification_quality([[1, 0], [0, 1]], labels, candidate_ids=CANDIDATES)
