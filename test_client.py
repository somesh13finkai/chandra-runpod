import requests
import sys
import json

# Runs directly on the pod (localhost) or from your Mac via the public proxy URL
API_URL = "http://localhost:8080/extract"

def test_api(s3_url: str, prompt: str = None):
    print(f"\n📤 Sending request for: {s3_url}")
    print(f"🌐 API endpoint: {API_URL}\n")

    payload = {"s3_url": s3_url}
    if prompt:
        payload["prompt"] = prompt

    try:
        response = requests.post(API_URL, json=payload, timeout=300)

        if response.status_code == 200:
            print("✅ Success! Extracted JSON:\n")
            print(json.dumps(response.json(), indent=2))
        else:
            print(f"❌ Failed with status {response.status_code}:")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not reach the FastAPI server.")
        print("   Make sure 'python3 main.py' is running on the pod.")
    except requests.exceptions.Timeout:
        print("❌ Timeout: The model took too long to respond (>300s).")
    except requests.exceptions.RequestException as e:
        print(f"❌ Unexpected error: {e}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 test_client.py s3://your-bucket/path/to/invoice.pdf")
        print("\nExample:")
        print("  python3 test_client.py s3://finkraft-invoices/hotel/invoice_001.pdf")
        sys.exit(1)

    s3_url = sys.argv[1]
    custom_prompt = sys.argv[2] if len(sys.argv) > 2 else None
    test_api(s3_url, custom_prompt)
