# test_app.py content
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi.testclient import TestClient
from app import app
import pytest

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "timestamp" in data
    assert "uptime" in data
    assert "model_version" in data

def test_predict_endpoint():
    sample_transaction = {
        "transaction_id": "test-789",
        "timestamp": "2025-03-22 09:15:30",
        "merchant_id": "MERCH00987",
        "customer_id": "CUST004567",
        "customer_location": "urban",
        "payment_amount": 150.75,
        "payment_method": "debit_card",
        "device_type": "web",
        "network_latency": 120.67,
        "result": "success",
        "latency_bin_encoded": 3,
        "network_latency_scaled": 0.189654,
        "merchant_rolling_avg_amount_scaled": 0.07895,
        "merchant_success_rate_scaled": 0.945678,
        "device_success_rate_scaled": 0.985432,
        "payment_method_rolling_success_scaled": 0.923456,
        "location_success_rate_scaled": 0.812345,
        "payment_location_success_rate_scaled": 0.876543,
        "merchant_transaction_count_log": 5.12345,
        "hourly_transaction_volume_log": 6.78901,
        'amount_log': 5.012,
        'merchant_rolling_avg_amount': 200.0,
        'hourly_transaction_volume': 1200,
        'merchant_success_rate': 0.97,
        'time_of_day': 'morning',
        'latency_bin': 'high',
        'day_name': 'Tuesday',
        'amount_bin': 'medium',
    }
    
    response = client.post("/predict", json=sample_transaction)
    assert response.status_code == 200
    data = response.json()
    assert data["transaction_id"] == "test-tx-123"
    assert "success_probability" in data
    assert isinstance(data["success_probability"], float)
    assert 0 <= data["success_probability"] <= 1
    assert "recommended_action" in data
    assert data["recommended_action"] in ["route", "review"]

def test_feedback_endpoint():
    feedback_data = {
        "transaction_id": "test-tx-123",
        "actual_success": True,
        "prediction_probability": 0.85
    }
    
    response = client.post("/feedback", json=feedback_data)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["message"] == "Feedback received"

if __name__ == "__main__":
    pytest.main()