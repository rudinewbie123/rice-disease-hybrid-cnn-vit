import numpy as np
from rice_disease.metrics import calculate_metrics


def test_metrics_fields_and_perfect_predictions():
    y=[0,1,2,0,1,2]; p=np.array([[.8,.1,.1],[.1,.8,.1],[.1,.1,.8],[.7,.2,.1],[.2,.7,.1],[.1,.2,.7]])
    result=calculate_metrics(y,p,["a","b","c"])
    assert result["accuracy"]==1.0 and result["macro_f1"]==1.0
    assert "macro_precision" in result and "macro_recall" in result
    assert len(result["confusion_matrix"])==3


def test_auc_not_reported_if_class_missing():
    result=calculate_metrics([0,0],np.array([[.9,.05,.05],[.8,.1,.1]]),["a","b","c"])
    assert result["roc_auc_ovr_macro"] is None and "roc_auc_note" in result
