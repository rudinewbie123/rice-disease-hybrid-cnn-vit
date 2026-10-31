"""Classification metrics and publication-ready evaluation plots."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def calculate_metrics(y_true, probabilities, class_names):
    """Calculate multiclass metrics; undefined class-wise AUCs are reported as None."""
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    names = list(class_names)
    n_classes = len(names)
    if n_classes < 2:
        raise ValueError("At least two class names are required.")
    if probs.ndim != 2 or probs.shape != (len(y), n_classes):
        raise ValueError(f"probabilities must have shape (n_samples, {n_classes}).")
    if len(y) == 0:
        raise ValueError("Cannot evaluate an empty target array.")
    if not np.isfinite(probs).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    if (probs < 0).any() or (probs > 1).any():
        raise ValueError("Probabilities must be in [0, 1].")
    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-4):
        raise ValueError("Each probability row must sum to 1.")
    if (y < 0).any() or (y >= n_classes).any():
        raise ValueError("y_true contains a class index outside class_names.")

    predicted = probs.argmax(axis=1)
    matrix = confusion_matrix(y, predicted, labels=list(range(n_classes)))
    report = classification_report(
        y, predicted, labels=list(range(n_classes)), target_names=names,
        zero_division=0, output_dict=True,
    )
    result = {
        "n_samples": int(len(y)),
        "class_names": names,
        "class_support": {name: int((y == i).sum()) for i, name in enumerate(names)},
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_precision": float(precision_score(y, predicted, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, predicted, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, predicted, average="macro", zero_division=0)),
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    binary = (y[:, None] == np.arange(n_classes)[None, :]).astype(int)
    per_class_auc = {}
    valid_auc = []
    for index, name in enumerate(names):
        positives = int(binary[:, index].sum())
        negatives = int(len(y) - positives)
        if positives == 0 or negatives == 0:
            per_class_auc[name] = None
        else:
            value = float(roc_auc_score(binary[:, index], probs[:, index]))
            per_class_auc[name] = value
            valid_auc.append(value)
    result["roc_auc_per_class_ovr"] = per_class_auc
    if len(valid_auc) == n_classes:
        result["roc_auc_ovr_macro"] = float(np.mean(valid_auc))
    else:
        result["roc_auc_ovr_macro"] = None
        result["roc_auc_note"] = (
            "Macro OvR ROC-AUC is undefined because at least one class has no "
            "positive or no negative samples in this evaluation split."
        )
    return result


def save_confusion_matrix(matrix, class_names, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(
        np.asarray(matrix), display_labels=class_names
    ).plot(ax=ax, cmap="Blues", colorbar=False, values_format="d")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_roc_curves(y_true, probabilities, class_names, path):
    """Save one-vs-rest ROC curves; omit classes without positive/negative support."""
    y = np.asarray(y_true, dtype=int)
    probs = np.asarray(probabilities, dtype=float)
    binary = (y[:, None] == np.arange(len(class_names))[None, :]).astype(int)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = 0
    auc_values = []
    for index, name in enumerate(class_names):
        target = binary[:, index]
        if target.min() == target.max():
            continue
        fpr, tpr, _ = roc_curve(target, probs[:, index])
        score = float(auc(fpr, tpr))
        auc_values.append(score)
        ax.plot(fpr, tpr, label=f"{name} (AUC={score:.3f})")
        plotted += 1
    ax.plot([0, 1], [0, 1], linestyle="--", label="Chance")
    ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False Positive Rate",
           ylabel="True Positive Rate", title="One-vs-Rest ROC Curves")
    if plotted:
        ax.legend(loc="lower right")
    else:
        ax.text(.5, .5, "ROC undefined: each class needs positive and negative samples",
                ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return {name: (float(auc(roc_curve(binary[:, i], probs[:, i])[0],
                                  roc_curve(binary[:, i], probs[:, i])[1]))
                   if binary[:, i].min() != binary[:, i].max() else None)
            for i, name in enumerate(class_names)}
