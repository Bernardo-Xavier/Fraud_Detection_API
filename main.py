import joblib
import pandas as pd
import uuid
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Dict, Any

# Import the CORS middleware
from fastapi.middleware.cors import CORSMiddleware

# --- 1. Application Setup & CORS Configuration ---
# Initialize the FastAPI application. This is the core of our microservice.
app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="A prototype API for assessing credit card transaction risk using a machine learning model.",
    version="1.0.0"
)

# Define the origins that are allowed to make requests to this API.
# This is the crucial fix to allow your Vue.js frontend to communicate with the API.
origins = ["*"]  # Allows all origins for development

# Add the CORS middleware to the application.
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)


# --- 2. Model and Scaler Loading ---
# On startup, the service loads the trained model and scaler from the artifact repository.
# This ensures that the model is in memory and ready to serve predictions without delay.
try:
    # Note: Using the model name from your provided code.
    model = joblib.load('./model/fraud_detection_model.joblib')
    scaler = joblib.load('./model/scaler.joblib')
    print("Model and scaler loaded successfully.")
except FileNotFoundError:
    print("FATAL ERROR: Model or scaler artifacts not found. The service cannot start.")
    print("Please run train_model.py to generate the required artifacts.")
    # In a real-world scenario, this would trigger a critical alert.
    model = None
    scaler = None

# --- 3. API Data Structures (Contracts) ---
# Pydantic models define the data schema for API requests and responses.
# This provides strong data validation and a clear, self-documenting contract for API consumers.

class Transaction(BaseModel):
    """
    Defines the structure for an incoming transaction request.
    The features V1-V28 are principal components from the original dataset.
    """
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

class PredictionResponse(BaseModel):
    """
    Defines the structure for the API's prediction response.
    """
    transaction_id: str
    risk_score: float = Field(..., description="The model's predicted probability of fraud (0 to 1).")
    is_fraud: bool = Field(..., description="A binary decision flag (true if fraud is suspected).")
    decision: str = Field(..., description="A human-readable risk decision (e.g., APPROVE, DECLINE, REVIEW).")
    confidence_level: float = Field(..., description="The model's confidence in its decision.")

# --- 4. API Endpoints ---

@app.get("/", summary="Health Check", description="Provides a basic health check of the API service.")
def read_root():
    """
    A simple health check endpoint. In a production environment, this would
    return the status of various system components (e.g., model loaded, database connection).
    """
    return {"status": "ok", "service": "Fraud Detection API", "model_ready": model is not None}


@app.post("/predict", response_model=PredictionResponse, summary="Assess Transaction Risk")
def predict_fraud(transaction: Transaction):
    """
    This endpoint processes a single transaction and returns a fraud risk assessment.

    - **Input:** A JSON object containing transaction features.
    - **Output:** A JSON object with the risk score, decision, and transaction ID.
    """
    if model is None or scaler is None:
        # This check prevents the service from trying to predict if the model failed to load.
        return {
            "error": "Model not loaded. Service is in a degraded state."
        }

    # Convert the incoming Pydantic model to a pandas DataFrame for processing.
    transaction_df = pd.DataFrame([transaction.dict()])
    
    # Ensure column order is correct, just in case
    feature_columns = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
                       'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
                       'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']
    transaction_df = transaction_df[feature_columns]

    # Apply the same scaling transformation used during training.
    transaction_scaled = scaler.transform(transaction_df)

    # Make the prediction. `predict_proba` returns the probability for each class.
    # We are interested in the probability of the "fraud" class (class 1).
    fraud_probability = model.predict_proba(transaction_scaled)[0][1]

    # Define risk thresholds for decisioning.
    is_fraud_decision = bool(fraud_probability > 0.5) 
    
    if fraud_probability > 0.75:
        decision_text = "DECLINE"
    elif fraud_probability > 0.5:
        decision_text = "REVIEW"
    else:
        decision_text = "APPROVE"

    # Generate a unique transaction ID for tracking and auditing purposes.
    trans_id = str(uuid.uuid4())

    # Calculate confidence level
    confidence = (fraud_probability if is_fraud_decision else 1 - fraud_probability)

    # Assemble the response payload.
    response = PredictionResponse(
        transaction_id=trans_id,
        risk_score=round(fraud_probability, 4),
        is_fraud=is_fraud_decision,
        decision=decision_text,
        confidence_level=round(confidence, 4)
    )

    return response

# This part allows for direct execution, but uvicorn in the Docker CMD is the primary way it runs.
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
