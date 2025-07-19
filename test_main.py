"""
Test configuration for LogPrompt API
"""

from typing import Any, Dict
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_root_endpoint() -> None:
    """Test the root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data: Dict[str, Any] = response.json()
    assert "message" in data
    assert "supported_models" in data


def test_models_endpoint() -> None:
    """Test the models listing endpoint"""
    response = client.get("/models")
    assert response.status_code == 200
    data: Dict[str, Any] = response.json()
    assert "supported_models" in data
    assert "loaded_models" in data


def test_health_endpoint() -> None:
    """Test the health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data: Dict[str, Any] = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_predict_endpoint() -> None:
    """Test the prediction endpoint"""
    test_request = {
        "text": "Hello world",
        "model_name": "bert-base-uncased",
        "task": "feature-extraction"
    }

    response = client.post("/predict", json=test_request)
    assert response.status_code == 200
    data: Dict[str, Any] = response.json()
    assert "model_name" in data
    assert "text" in data


def test_invalid_model() -> None:
    """Test prediction with invalid model"""
    test_request = {
        "text": "Hello world",
        "model_name": "invalid-model",
        "task": "feature-extraction"
    }

    response = client.post("/predict", json=test_request)
    assert response.status_code == 400
    data: Dict[str, Any] = response.json()
    assert "detail" in data
    assert "invalid-model not supported" in data["detail"]
