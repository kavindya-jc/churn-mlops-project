# src/evaluate.py

import pandas as pd
import joblib
import mlflow
import json
import os
import logging
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score
)

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate_best_model():
    logger.info("Loading best model and test data...")

    model = joblib.load("models/best_model.pkl")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob)
    }

    print("\n" + "="*50)
    print("📊 BEST MODEL EVALUATION RESULTS")
    print("="*50)
    for k, v in metrics.items():
        print(f"  {k:12s}: {v:.4f}")

    # Save metrics to JSON for DVC to track
    os.makedirs("reports", exist_ok=True)
    with open("reports/metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)
    logger.info("✅ Metrics saved to reports/metrics.json")

    # Log to MLflow
    mlflow.set_experiment("churn_prediction")
    with mlflow.start_run(run_name="BestModel_Evaluation"):
        mlflow.log_metrics(metrics)
        logger.info("✅ Evaluation metrics logged to MLflow!")

    return metrics


if __name__ == "__main__":
    evaluate_best_model()
