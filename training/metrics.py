# training/metrics.py
from __future__ import annotations
import numpy as np

def build_compute_metrics():
    """
    Returns a compute_metrics function compatible with HF Trainer.
    Uses evaluate.load inside to avoid import issues at module import time.
    """
    import evaluate

    f1_metric = evaluate.load("f1")
    acc_metric = evaluate.load("accuracy")

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=1)
        f1 = f1_metric.compute(predictions=preds, references=labels, average="macro")["f1"]
        acc = acc_metric.compute(predictions=preds, references=labels)["accuracy"]
        return {"f1": f1, "accuracy": acc}

    return compute_metrics