# airflow_dags/churn_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import os
import sys
import logging

# Add project root to path so we can import our scripts
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# DEFAULT ARGUMENTS
# ─────────────────────────────────────────────
default_args = {
    'owner': 'chalani',
    'depends_on_past': False,
    'start_date': datetime(2024, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5)
}

# ─────────────────────────────────────────────
# TASK FUNCTIONS
# ─────────────────────────────────────────────


def task_data_ingestion():
    """Task 1: Load raw data and verify it"""
    logger.info("="*50)
    logger.info("TASK 1: DATA INGESTION STARTED")
    logger.info("="*50)

    filepath = "data/raw/telco_churn.csv"

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    df = pd.read_csv(filepath)

    assert df.shape[0] > 0, "Dataset is empty!"
    assert df.shape[1] == 21, f"Expected 21 columns, got {df.shape[1]}"

    logger.info(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return "Data ingestion complete"


def task_data_validation():
    """Task 2: Validate data quality"""
    logger.info("="*50)
    logger.info("TASK 2: DATA VALIDATION STARTED")
    logger.info("="*50)

    df = pd.read_csv("data/raw/telco_churn.csv")

    # Check required columns exist
    required_cols = [
        'customerID', 'gender', 'SeniorCitizen', 'Partner',
        'Dependents', 'tenure', 'PhoneService', 'MonthlyCharges',
        'TotalCharges', 'Churn'
    ]
    for col in required_cols:
        assert col in df.columns, f"Missing column: {col}"

    # Check target column has correct values
    assert set(df['Churn'].unique()) == {'Yes', 'No'}, \
        "Churn column has unexpected values!"

    # Check no negative tenure
    assert df['tenure'].min() >= 0, "Negative tenure found!"

    logger.info("✅ All validation checks passed!")
    return "Data validation complete"


def task_feature_engineering():
    """Task 3: Preprocess and engineer features"""
    logger.info("="*50)
    logger.info("TASK 3: FEATURE ENGINEERING STARTED")
    logger.info("="*50)

    from src.preprocessing import (
        handle_missing_values,
        encode_categorical,
        scale_features,
        split_data,
        save_processed_data
    )
    from src.data_ingestion import load_data

    df = load_data("data/raw/telco_churn.csv")
    df = handle_missing_values(df)
    df = encode_categorical(df)
    X, y, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)

    logger.info("✅ Feature engineering complete!")
    return "Feature engineering complete"


def task_model_training():
    """Task 4: Train all models"""
    logger.info("="*50)
    logger.info("TASK 4: MODEL TRAINING STARTED")
    logger.info("="*50)

    import mlflow
    from src.train import (
        load_processed_data,
        train_and_log
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier

    X_train, X_test, y_train, y_test = load_processed_data()
    mlflow.set_experiment("churn_prediction_airflow")

    # Train all 3 models
    lr_params = {"C": 1.0, "max_iter": 1000, "solver": "lbfgs"}
    rf_params = {"n_estimators": 100, "max_depth": 10, "min_samples_split": 5}
    xgb_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1}

    lr_auc, lr = train_and_log(
        LogisticRegression(**lr_params, random_state=42),
        "LR_Airflow", lr_params, X_train, X_test, y_train, y_test
    )
    rf_auc, rf = train_and_log(
        RandomForestClassifier(**rf_params, random_state=42),
        "RF_Airflow", rf_params, X_train, X_test, y_train, y_test
    )
    xgb_auc, xgb = train_and_log(
        XGBClassifier(**xgb_params, random_state=42, eval_metric='logloss'),
        "XGB_Airflow", xgb_params, X_train, X_test, y_train, y_test
    )

    logger.info("✅ All models trained!")
    return "Model training complete"


def task_model_evaluation():
    """Task 5: Evaluate best model"""
    logger.info("="*50)
    logger.info("TASK 5: MODEL EVALUATION STARTED")
    logger.info("="*50)

    from src.evaluate import evaluate_best_model
    metrics = evaluate_best_model()

    logger.info(f"✅ Evaluation complete! ROC-AUC: {metrics['roc_auc']:.4f}")
    return "Model evaluation complete"


def task_model_registration():
    """Task 6: Register best model to MLflow"""
    logger.info("="*50)
    logger.info("TASK 6: MODEL REGISTRATION STARTED")
    logger.info("="*50)

    import mlflow
    import joblib

    model = joblib.load("models/best_model.pkl")

    mlflow.set_experiment("churn_prediction_airflow")
    with mlflow.start_run(run_name="Model_Registration"):
        mlflow.sklearn.log_model(
            model,
            artifact_path="registered_model",
            registered_model_name="ChurnPredictionModel"
        )
        logger.info("✅ Model registered in MLflow!")

    return "Model registration complete"


# ─────────────────────────────────────────────
# DAG DEFINITION
# ─────────────────────────────────────────────
with DAG(
    dag_id='churn_prediction_pipeline',
    default_args=default_args,
    description='End-to-end churn prediction MLOps pipeline',
    schedule='@weekly',
    catchup=False,
    tags=['churn', 'mlops']
) as dag:

    # Define tasks
    t1 = PythonOperator(
        task_id='data_ingestion',
        python_callable=task_data_ingestion
    )

    t2 = PythonOperator(
        task_id='data_validation',
        python_callable=task_data_validation
    )

    t3 = PythonOperator(
        task_id='feature_engineering',
        python_callable=task_feature_engineering
    )

    t4 = PythonOperator(
        task_id='model_training',
        python_callable=task_model_training
    )

    t5 = PythonOperator(
        task_id='model_evaluation',
        python_callable=task_model_evaluation
    )

    t6 = PythonOperator(
        task_id='model_registration',
        python_callable=task_model_registration
    )

    # Set task dependencies (order)
    t1 >> t2 >> t3 >> t4 >> t5 >> t6
