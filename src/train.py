# src/train.py

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_processed_data():
    logger.info("Loading processed data...")
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").squeeze()
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()
    logger.info(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def save_confusion_matrix(y_test, y_pred, model_name):
    os.makedirs("reports", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    path = f"reports/confusion_matrix_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path


def save_roc_curve(y_test, y_pred_prob, model_name):
    os.makedirs("reports", exist_ok=True)
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend()
    path = f"reports/roc_curve_{model_name}.png"
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    return path


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_prob)
    }
    logger.info(f"\n📊 {model_name} Results:")
    for k, v in metrics.items():
        logger.info(f"   {k}: {v:.4f}")
    return metrics, y_pred, y_pred_prob


def train_and_log(model, model_name, params, X_train, X_test, y_train, y_test):
    logger.info(f"\n🚀 Training {model_name}...")
    with mlflow.start_run(run_name=model_name):
        model.fit(X_train, y_train)
        metrics, y_pred, y_pred_prob = evaluate_model(
            model, X_test, y_test, model_name
        )
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        cm_path = save_confusion_matrix(y_test, y_pred, model_name)
        roc_path = save_roc_curve(y_test, y_pred_prob, model_name)
        mlflow.log_artifact(cm_path)
        mlflow.log_artifact(roc_path)
        mlflow.sklearn.log_model(model, artifact_path=model_name)
        logger.info(f"✅ {model_name} logged to MLflow!")
    return metrics["roc_auc"], model


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_processed_data()

    mlflow.set_experiment("churn_prediction")

    # Model 1: Logistic Regression
    lr_params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    lr_auc, lr = train_and_log(
        LogisticRegression(**lr_params, random_state=42),
        "LogisticRegression", lr_params,
        X_train, X_test, y_train, y_test
    )

    # Model 2: Random Forest
    rf_params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}
    rf_auc, rf = train_and_log(
        RandomForestClassifier(**rf_params, random_state=42),
        "RandomForest", rf_params,
        X_train, X_test, y_train, y_test
    )

    # Model 3: XGBoost
    xgb_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}
    xgb_auc, xgb = train_and_log(
        XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss'),
        "XGBoost", xgb_params,
        X_train, X_test, y_train, y_test
    )

    # Pick best model
    results = {
        "LogisticRegression": (lr_auc, lr),
        "RandomForest": (rf_auc, rf),
        "XGBoost": (xgb_auc, xgb)
    }

    best_name = max(results, key=lambda k: results[k][0])
    best_model = results[best_name][1]

    print("\n" + "="*50)
    print("🏆 MODEL COMPARISON (ROC-AUC)")
    print("="*50)
    for name, (auc, _) in results.items():
        marker = " ← BEST" if name == best_name else ""
        print(f"  {name:25s}: {auc:.4f}{marker}")

    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"\n✅ Best model saved: models/best_model.pkl")
    print(f"✅ MLflow experiments logged!")
    print(f"\n💡 To view MLflow UI run:  mlflow ui")
