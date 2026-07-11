from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "metrics"
metrics_file = OUTPUT_PATH / "test_metrics.csv"


def evaluate(truth, predictions, MODEL_NAME, feature_type, vectorizer, total_latency, num_samples):

    accuracy = accuracy_score(truth, predictions)
    precision = precision_score(truth, predictions)
    recall = recall_score(truth, predictions)
    f1 = f1_score(truth, predictions)

    tn, fp, fn, tp = confusion_matrix(truth, predictions).ravel()

    throughput = num_samples / total_latency if total_latency else None # samples per second
    avg_latency = total_latency / num_samples if num_samples else None

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    # Append metrics
    metrics_df = pd.DataFrame([{
        "feature_type": feature_type,                
        "model_name": MODEL_NAME,
        "vectorizer": vectorizer,
        "best_params": "",                     
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "throughput": throughput,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "total_latency_sec": total_latency,
        "avg_latency_per_sample_sec": avg_latency,
        "num_samples": num_samples,
    }])

    metrics_df.to_csv(
        metrics_file,
        mode="a",
        header=not metrics_file.exists(),
        index=False,
    )

    print(f"Metrics appended.")

    return predictions