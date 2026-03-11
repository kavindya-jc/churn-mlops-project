# src/preprocessing.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import os
import logging
import joblib

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fixes missing and incorrect values in the dataset.
    """
    logger.info("Handling missing values...")

    # TotalCharges is stored as string, needs to be numeric
    # Empty strings " " exist in TotalCharges — replace with NaN first
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # Fill missing TotalCharges with median value
    median_val = df['TotalCharges'].median()
    df['TotalCharges'] = df['TotalCharges'].fillna(median_val)

    logger.info(f"Missing values after fix:\n{df.isnull().sum()}")
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts text columns into numbers so ML models can understand them.
    """
    logger.info("Encoding categorical variables...")

    # Drop customerID - it's just an identifier, not useful for prediction
    df = df.drop('customerID', axis=1)

    # gender: Male=1, Female=0
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

    # Simple Yes/No columns → 1/0
    yes_no_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})

    # Target column Churn: Yes=1, No=0
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Multi-category columns → One Hot Encoding
    multi_cols = [
        'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport',
        'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
    ]
    df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

    logger.info(f"Shape after encoding: {df.shape}")
    return df


def scale_features(df: pd.DataFrame):
    """
    Scales numeric features so they're on the same scale.
    Returns scaled data + the scaler object (to use later for predictions).
    """
    logger.info("Scaling numeric features...")

    # Separate features (X) and target (y)
    X = df.drop('Churn', axis=1)
    y = df['Churn']

    # Columns to scale
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    scaler = StandardScaler()
    X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

    logger.info("Scaling complete!")
    return X, y, scaler


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Splits data into training set and testing set.
    80% for training, 20% for testing.
    """
    logger.info(
        f"Splitting data: {int((1-test_size)*100)}% train, {int(test_size*100)}% test")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    logger.info(f"Training set: {X_train.shape[0]} rows")
    logger.info(f"Testing set:  {X_test.shape[0]} rows")

    return X_train, X_test, y_train, y_test


def save_processed_data(X_train, X_test, y_train, y_test, scaler):
    """
    Saves the processed data and scaler to disk.
    """
    logger.info("Saving processed data...")

    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # Save as CSV
    X_train.to_csv("data/processed/X_train.csv", index=False)
    X_test.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)

    # Save scaler (needed later for API predictions)
    joblib.dump(scaler, "models/scaler.pkl")

    logger.info("✅ All processed data saved to data/processed/")
    logger.info("✅ Scaler saved to models/scaler.pkl")


if __name__ == "__main__":
    from data_ingestion import load_data

    # Run full preprocessing pipeline
    df = load_data("data/raw/telco_churn.csv")
    df = handle_missing_values(df)
    df = encode_categorical(df)
    X, y, scaler = scale_features(df)
    X_train, X_test, y_train, y_test = split_data(X, y)
    save_processed_data(X_train, X_test, y_train, y_test, scaler)

    print("\n✅ Preprocessing complete! Check data/processed/ folder")
