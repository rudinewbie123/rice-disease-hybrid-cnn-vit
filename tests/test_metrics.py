import numpy as np
import pytest

from rice_disease.metrics import calculate_metrics


def _perfect_probabilities():
    return np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],
                     [.7,.2,.1],[.2,.7,.1],[.1,.2,.7]])


def test_metrics_fields_and_perfect_predictions():
    result = calculate_metrics([0,1,2,0,1,2], _perfect_probabilities(), ["a","b","c"])
    assert result["accuracy"] == 1.0 and result["macro_f1"] == 1.0
    assert result["macro_precision"] == 1.0 and result["macro_recall"] == 1.0
    assert result["roc_auc_ovr_macro"] == 1.0
    assert all(value == 1.0 for value in result["roc_auc_per_class_ovr"].values())
    assert len(result["confusion_matrix"]) == 3


def test_auc_not_reported_if_class_missing():
    result = calculate_metrics([0,0], np.array([[.9,.05,.05],[.8,.1,.1]]), ["a","b","c"])
    assert result["roc_auc_ovr_macro"] is None
    assert result["roc_auc_per_class_ovr"]["a"] is None


@pytest.mark.parametrize("probabilities", [
    np.array([[.5,.5],[.2,.8]]),
    np.array([[.5,.5,.0],[.2,.8,.0]]),
    np.array([[.5,.5,np.nan],[.2,.8,0.0]]),
])
def test_invalid_probabilities_raise(probabilities):
    with pytest.raises(ValueError):
        calculate_metrics([0,1], probabilities, ["a","b","c"])
