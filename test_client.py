import base64
import requests
import sys

# RunPod automatically creates a public URL for your exposed ports.
# Your pod ID is olxeq34clih41y, and your FastAPI server is on port 8080.
API_URL = "https://olxeq34clih41y-8080.proxy.runpod.net/extract"

def test_api(image_path):
    print(f"Loading image {image_path}...")
    try:
        with open(image_path, "rb") as f:
            base64_image = base64.b64encode(f.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Error: Could not find image at {image_path}")
        print("Please provide a valid path to an invoice image.")
        return

    # Auto-detect file type and prepend the proper data URI
    if image_path.lower().endswith(".pdf"):
        prefix = "data:application/pdf;base64,"
    elif image_path.lower().endswith(".png"):
        prefix = "data:image/png;base64,"
    else:
        prefix = "data:image/jpeg;base64,"
        
    full_data_uri = prefix + base64_image

    payload = {
        "base64_image": full_data_uri,
        "prompt": "Extract the total amount, date, and vendor from this invoice."
    }

    print(f"Sending request to {API_URL}...")
    try:
        response = requests.post(API_URL, json=payload, timeout=120)
        
        if response.status_code == 200:
            print("\n✅ Success! Here is the extracted JSON:")
            print(response.json())
        else:
            print(f"\n❌ Failed with status {response.status_code}:")
            print(response.text)
            
    except requests.exceptions.RequestException as e:
        print(f"\n❌ Connection Error: {e}")
        print("Make sure port 8080 is exposed in your RunPod template settings!")

if __name__ == "__main__":
    # You can pass an image path as an argument, or change this default
    default_image = "invoice.png" 
    
    if len(sys.argv) > 1:
        test_api(sys.argv[1])
    else:
        test_api(default_image)
