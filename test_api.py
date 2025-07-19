#!/usr/bin/env python3
"""
Test script for LogPrompt API
"""

from typing import Any, Dict
import requests

# API base URL
BASE_URL = "http://localhost:8000"


def test_api() -> None:
    """Test the API endpoints"""

    # Test root endpoint
    print("Testing root endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

    # Test models endpoint
    print("Testing models endpoint...")
    response = requests.get(f"{BASE_URL}/models")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

    # Test health endpoint
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    print()

    # Test prediction with BERT base
    print("Testing prediction with BERT base...")
    test_request: Dict[str, Any] = {
        "text": "Hello, this is a test sentence for BERT processing.",
        "model_name": "bert-base-uncased",
        "task": "feature-extraction"
    }

    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_request,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status: {response.status_code}")
    result: Dict[str, Any] = response.json()

    if "embeddings" in result and result["embeddings"]:
        print(f"Model: {result['model_name']}")
        print(f"Text: {result['text']}")
        embeddings_shape = (
            f"{len(result['embeddings'])} x {len(result['embeddings'][0])}"
        )
        print(f"Embeddings shape: {embeddings_shape}")
    else:
        print(f"Response: {result}")
    print()

    # Test with RoBERTa
    print("Testing prediction with RoBERTa...")
    test_request["model_name"] = "roberta-base"

    response = requests.post(
        f"{BASE_URL}/predict",
        json=test_request,
        headers={"Content-Type": "application/json"}
    )

    print(f"Status: {response.status_code}")
    result = response.json()

    if "embeddings" in result and result["embeddings"]:
        print(f"Model: {result['model_name']}")
        print(f"Text: {result['text']}")
        embeddings_shape = (
            f"{len(result['embeddings'])} x {len(result['embeddings'][0])}"
        )
        print(f"Embeddings shape: {embeddings_shape}")
    else:
        print(f"Response: {result}")


if __name__ == "__main__":
    try:
        test_api()
    except requests.exceptions.ConnectionError:
        print(
            "Error: Could not connect to API. "
            "Make sure the server is running on http://localhost:8000"
        )
    except Exception as e:
        print(f"Error: {e}")
