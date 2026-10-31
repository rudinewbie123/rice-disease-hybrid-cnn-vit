from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (accuracy_score,classification_report,confusion_matrix,f1_score,
    precision_score,recall_score,roc_auc_score,ConfusionMatrixDisplay)
from sklearn.preprocessing import label_binarize


def calculate_metrics(y_true,probabilities,class_names):
    y=np.asarray(y_true); p=np.asarray(probabilities); pred=p.argmax(1)
    result={"accuracy":float(accuracy_score(y,pred)),
      "macro_precision":float(precision_score(y,pred,average="macro",zero_division=0)),
      "macro_recall":float(recall_score(y,pred,average="macro",zero_division=0)),
      "macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),
      "classification_report":classification_report(y,pred,labels=list(range(len(class_names))),
          target_names=class_names,zero_division=0,output_dict=True),
      "confusion_matrix":confusion_matrix(y,pred,labels=list(range(len(class_names)))).tolist()}
    try:
        binary=label_binarize(y,classes=list(range(len(class_names))))
        if binary.shape[1]==len(class_names) and all(binary.sum(0)>0):
            result["roc_auc_ovr_macro"]=float(roc_auc_score(binary,p,average="macro",multi_class="ovr"))
        else:
            result["roc_auc_ovr_macro"]=None; result["roc_auc_note"]="ROC-AUC requires positive support for every class."
    except ValueError as exc:
        result["roc_auc_ovr_macro"]=None; result["roc_auc_note"]=str(exc)
    return result


def save_confusion_matrix(matrix,class_names,path):
    fig,ax=plt.subplots(figsize=(7,6))
    ConfusionMatrixDisplay(np.asarray(matrix),display_labels=class_names).plot(ax=ax,cmap="Blues",colorbar=False)
    ax.set_title("Confusion Matrix"); fig.tight_layout(); fig.savefig(Path(path),dpi=200); plt.close(fig)
