# src/data_ingestion.py

import pandas as pd
import os
import logging

# Set up logging - this prints helpful messages as the script runs
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(filepath: str) -> pd.DataFrame:
    """
    Loads raw CSV data from the given filepath.
    Returns a pandas DataFrame.
    """
    logger.info(f"Loading data from: {filepath}")

    # Check if file exists
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at: {filepath}")

    df = pd.read_csv(filepath)

    logger.info(f"Data loaded successfully!")
    logger.info(f"Shape: {df.shape}")          # rows x columns
    logger.info(f"Columns: {list(df.columns)}")

    return df


def basic_eda(df: pd.DataFrame):
    """
    Prints basic information about the dataset.
    EDA = Exploratory Data Analysis
    """
    print("\n" + "="*50)
    print("BASIC DATA EXPLORATION")
    print("="*50)

    print(f"\n📊 Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")

    print(f"\n📋 Column Names:\n{list(df.columns)}")

    print(f"\n🔍 Data Types:\n{df.dtypes}")

    print(f"\n❓ Missing Values:\n{df.isnull().sum()}")

    print(f"\n📈 Churn Distribution:\n{df['Churn'].value_counts()}")
    print(
        f"Churn Rate: {df['Churn'].value_counts(normalize=True)['Yes']*100:.2f}%")

    print(f"\n📉 Numeric Summary:\n{df.describe()}")


if __name__ == "__main__":
    # This runs when you execute the script directly
    RAW_DATA_PATH = "data/raw/telco_churn.csv"

    df = load_data(RAW_DATA_PATH)
    basic_eda(df)
