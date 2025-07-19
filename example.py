#!/usr/bin/env python3
"""
Example usage of LogPrompt API
"""

import requests
import time


def main() -> None:
    base_url = "http://localhost:8000"

    print("🚀 LogPrompt API Example")
    print("=" * 50)

    # Check if API is running
    try:
        response = requests.get(f"{base_url}/health")
        if response.status_code == 200:
            print("✅ API is running!")
        else:
            print("❌ API is not responding correctly")
            return
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Please make sure it's running.")
        print("Run: make run")
        return

    # Example texts for testing
    test_texts = [
        "Hello, this is a test sentence for BERT processing.",
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming the world.",
        "Natural language processing enables computers to understand text."
    ]

    # Test different models
    models_to_test = [
        "bert-base-uncased",
        "roberta-base"
    ]

    for model_name in models_to_test:
        print(f"\n🤖 Testing model: {model_name}")
        print("-" * 30)
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text[:50]}...")
            request_data = {
                "text": text,
                "model_name": model_name,
                "task": "feature-extraction"
            }
            start_time = time.time()
            try:
                response = requests.post(
                    f"{base_url}/predict",
                    json=request_data,
                    timeout=60
                )
                end_time = time.time()
                if response.status_code == 200:
                    result = response.json()
                    if "embeddings" in result and result["embeddings"]:
                        embeddings = result["embeddings"]
                        print(
                            f"✅ Success! Shape: {len(embeddings)} x {len(embeddings[0])}")
                        print(f"⏱️  Time: {end_time - start_time:.2f}s")
                    else:
                        print(f"⚠️  Response: {result}")
                else:
                    print(f"❌ Error: {response.status_code}")
                    print(f"Response: {response.text}")
            except requests.exceptions.Timeout:
                print("⏱️  Request timed out")
            except Exception as e:
                print(f"❌ Error: {e}")
    print("\n" + "=" * 50)
    print("🎉 Testing completed!")
    print("📚 API Documentation: http://localhost:8000/docs")


if __name__ == "__main__":
    main()
