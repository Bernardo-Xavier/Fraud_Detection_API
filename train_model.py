# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib
import os

# --- 1. System Configuration & Constants ---
# In a production banking system, these paths would be managed by a centralized configuration service
# or environment variables for security and portability.
DATA_PATH = 'creditcard.csv'
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'fraud_detection_model.joblib')
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.joblib')

# --- 2. Data Ingestion and Preparation ---
# This phase simulates the ETL (Extract, Transform, Load) process common in banking data warehouses.
def load_and_prepare_data(path):
    """
    Loads transaction data, performs basic validation, and prepares it for modeling.
    
    Args:
        path (str): The path to the CSV data file.

    Returns:
        pandas.DataFrame: A preprocessed DataFrame ready for feature engineering.
    """
    print("Initiating data ingestion protocol...")
    try:
        df = pd.read_csv(path)
    except FileNotFoundError:
        print(f"FATAL ERROR: Data asset not found at {path}. System halt.")
        print("Please download the dataset from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        exit()

    # Basic data integrity check
    if 'Class' not in df.columns or 'Amount' not in df.columns:
        raise ValueError("Data asset integrity compromised. Required columns ('Class', 'Amount') are missing.")
    
    print("Data ingestion successful. Asset integrity validated.")
    return df

# --- 3. Feature Engineering and Model Training ---
# This section represents the core modeling work performed by a Quantitative Analytics or Data Science team.
def train_fraud_detection_model(df):
    """
    Engineers features, trains a classifier, and evaluates its performance.

    Args:
        df (pandas.DataFrame): The preprocessed transaction data.
    """
    print("Commencing model training workflow...")

    # Define features (X) and the target variable (y)
    # 'Class' is the ground truth label: 1 for fraud, 0 for legitimate.
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Splitting the data into training and testing sets is a standard protocol for model validation.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Feature Scaling: Standardizing feature values is crucial for an optimal outcome.
    # The 'Time' and 'Amount' columns have vastly different scales than the PCA components (V1-V28).
    print("Applying feature scaling transformation...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model Selection: Logistic Regression is chosen for its interpretability and efficiency,
    # making it a strong baseline model in many financial applications.
    # `class_weight='balanced'` addresses the severe class imbalance in fraud datasets.
    print("Training Logistic Regression classifier...")
    model = LogisticRegression(class_weight='balanced', solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Model training complete.")

    # --- 4. Model Performance Audit ---
    # Before deployment, any model must pass a rigorous performance audit.
    print("\n--- Model Performance Audit ---")
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    print("-----------------------------\n")

    # --- 5. Model Serialization ---
    # The trained model and scaler are serialized to disk for deployment into the real-time API.
    # This process is analogous to packaging a model for production release.
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
        print(f"Created model artifact directory: {MODEL_DIR}")

    joblib.dump(model, MODEL_PATH)
    print(f"Model artifact successfully saved to: {MODEL_PATH}")
    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler artifact successfully saved to: {SCALER_PATH}")


if __name__ == "__main__":
    transaction_data = load_and_prepare_data(DATA_PATH)
    train_fraud_detection_model(transaction_data)
    print("\nSystem workflow concluded. Model is ready for operational deployment.")
