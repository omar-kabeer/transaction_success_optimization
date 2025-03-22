# app.py content
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import joblib
import pandas as pd
import numpy as np
import time
import uuid
import logging
from datetime import datetime
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load model and pipeline
MODEL_PATH = "models/Optimized XGBoost.pkl"
PIPELINE_PATH = "models/feature_engineering_pipeline.pkl"

model = joblib.load(MODEL_PATH)
preprocessing_pipeline = joblib.load(PIPELINE_PATH)

# Initialize FastAPI app
app = FastAPI(
    title="Transaction Success Predictor API",
    description="API for predicting transaction success probability",
    version="1.0.0"
)

# Define data models
class TransactionData(BaseModel):
    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: str = Field(..., description="Timestamp of the transaction in YYYY-MM-DD HH:MM:SS format")
    merchant_id: str = Field(..., description="Unique identifier for the merchant")
    customer_id: str = Field(..., description="Unique identifier for the customer")
    customer_location: str = Field(..., description="Location type of the customer (urban, suburban, etc.)")
    payment_amount: float = Field(..., description="Transaction payment amount")
    payment_method: str = Field(..., description="Payment method used (credit_card, bank_transfer, etc.)")
    device_type: str = Field(..., description="Type of device used for transaction (mobile, web, etc.)")
    network_latency: float = Field(..., description="Latency in milliseconds for transaction processing")
    result: str = Field(..., description="Transaction result (success, pending, failed)")
    
    latency_bin_encoded: int = Field(..., description="Encoded category of network latency")
    network_latency_scaled: float = Field(..., description="Scaled network latency value")
    merchant_rolling_avg_amount_scaled: float = Field(..., description="Scaled rolling average transaction amount for the merchant")
    merchant_success_rate_scaled: float = Field(..., description="Scaled success rate of the merchant")
    device_success_rate_scaled: float = Field(..., description="Scaled success rate of transactions per device")
    payment_method_rolling_success_scaled: float = Field(..., description="Scaled rolling success rate for the payment method")
    location_success_rate_scaled: float = Field(..., description="Scaled success rate of transactions per customer location")
    payment_location_success_rate_scaled: float = Field(..., description="Scaled success rate of transactions per payment location")
    
    merchant_transaction_count_log: float = Field(..., description="Log-transformed count of transactions for the merchant")
    hourly_transaction_volume_log: float = Field(..., description="Log-transformed transaction volume per hour")
    amount_log: float = Field(..., description="Log-transformed transaction amount")
    merchant_rolling_avg_amount: float = Field(..., description="Rolling average transaction amount for the merchant")
    hourly_transaction_volume: int = Field(..., description="Total transaction volume per hour")
    merchant_success_rate: float = Field(..., description="Overall success rate of the merchant")
    
    time_of_day: str = Field(..., description="Time of the day category (morning, afternoon, evening, etc.)")
    latency_bin: str = Field(..., description="Latency bin category (low, medium, high)")
    day_name: str = Field(..., description="Name of the day (Monday, Tuesday, etc.)")
    amount_bin: str = Field(..., description="Transaction amount category (low, medium, high)")

    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "test-456",
                "timestamp": "2025-03-21 14:30:45",
                "merchant_id": "MERCH00123",
                "customer_id": "CUST002345",
                "customer_location": "suburban",
                "payment_amount": 75.50,
                "payment_method": "credit_card",
                "device_type": "mobile",
                "network_latency": 98.45,
                "result": "pending",
                "latency_bin_encoded": 2,
                "network_latency_scaled": 0.142367,
                "merchant_rolling_avg_amount_scaled": 0.05672,
                "merchant_success_rate_scaled": 0.912345,
                "device_success_rate_scaled": 0.974562,
                "payment_method_rolling_success_scaled": 0.899874,
                "location_success_rate_scaled": 0.765432,
                "payment_location_success_rate_scaled": 0.832145,
                "merchant_transaction_count_log": 4.56789,
                "hourly_transaction_volume_log": 6.12345,
                "amount_log": 4.321,
                "merchant_rolling_avg_amount": 100.0,
                "hourly_transaction_volume": 1000,
                "merchant_success_rate": 0.95,
                "time_of_day": "afternoon",
                "latency_bin": "medium",
                "day_name": "Monday",
                "amount_bin": "low"
            }
        }

class PredictionResponse(BaseModel):
    transaction_id: str
    success_probability: float
    recommended_action: str
    prediction_time: str
    model_version: str

class FeedbackData(BaseModel):
    transaction_id: str
    actual_success: bool
    prediction_probability: float
    
    class Config:
        json_schema_extra = {
            "example": {
                "transaction_id": "tx-12345",
                "actual_success": True,
                "prediction_probability": 0.87
            }
        }

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    uptime: float
    model_version: str

# Track API start time
start_time = time.time()
feedback_data = []

# Helper functions
def preprocess_transaction_data(transaction_data):
    """Preprocess transaction data using the pipeline"""
    if isinstance(transaction_data, dict):
        transaction_df = pd.DataFrame([transaction_data])
    else:
        transaction_df = pd.DataFrame([transaction_data.dict()])
    
    # Convert to expected format
    return preprocessing_pipeline.fit_transform(transaction_df)

def log_feedback(feedback: FeedbackData):
    """Log feedback data for later model retraining"""
    feedback_data.append(feedback.dict())
    
    # If we've collected enough feedback, save to disk
    if len(feedback_data) >= 100:
        df = pd.DataFrame(feedback_data)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        df.to_csv(f"./feedback/feedback_data_{timestamp}.csv", index=False)
        feedback_data.clear()
    
    logger.info(f"Feedback logged for transaction {feedback.transaction_id}")

# API endpoints
app = FastAPI()

# Get absolute path
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Ensure directory exists
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

# Mount the "static" directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/")
def read_root():
    return {"message": "Welcome to the Transaction Success Predictor API"}

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    favicon_path = os.path.join(static_dir, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    return {"error": "favicon.ico not found"}

@app.get("/", response_model=dict)
async def root():
    return {"message": "Welcome to the Transaction Success Predictor API"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(transaction: TransactionData):
    """
    Predict transaction success probability
    """
    try:
        # Preprocess data
        preprocessed_data = preprocess_transaction_data(transaction)
        
        # Make prediction
        success_probability = float(model.predict_proba(preprocessed_data)[:, 1][0])
        
        # Determine recommended action
        recommended_action = "route" if success_probability > 0.5 else "review"
        
        # Prepare response
        response = {
            "transaction_id": transaction.transaction_id,
            "success_probability": success_probability,
            "recommended_action": recommended_action,
            "prediction_time": datetime.now().isoformat(),
            "model_version": "1.0.0"
        }
        
        logger.info(f"Prediction made for transaction {transaction.transaction_id}: {success_probability:.4f}")
        return response
        
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/feedback")
async def feedback(feedback_data: FeedbackData, background_tasks: BackgroundTasks):
    """
    Log transaction outcome feedback for continuous learning
    """
    try:
        # Process feedback in the background to not block the API
        background_tasks.add_task(log_feedback, feedback_data)
        return {"status": "success", "message": "Feedback received"}
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Feedback processing error: {str(e)}")

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Check if the API is running correctly
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "uptime": time.time() - start_time,
        "model_version": "1.0.0"
    }

# Run the API with uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)