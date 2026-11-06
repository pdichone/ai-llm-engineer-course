import requests
import json


def test_api():
    # API endpoint
    base_url = "http://localhost:8000"

    # Test single prediction
    single_review = {
        "text": "This movie was absolutely amazing! Great performance by all actors."
    }

    print("\nTesting single prediction...")
    response = requests.post(f"{base_url}/predict", json=single_review)
    print(json.dumps(response.json(), indent=2))

    # Test batch prediction
    batch_reviews = {
        "texts": [
            "This movie was absolutely amazing! Great performance by all actors.",
            "Terrible movie, waste of time and money.",
            "Pretty average movie, nothing special.",
            "It was okay. I've seen better.",
        ]
    }

    print("\nTesting batch prediction...")
    response = requests.post(f"{base_url}/predict_batch", json=batch_reviews)
    print(json.dumps(response.json(), indent=2))


if __name__ == "__main__":
    test_api()
