import json
import base64
import mimetypes
import os
import httpx
import boto3
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from urllib.parse import urlparse

app = FastAPI(title="Serverless GPU API")

# S3 client - credentials are injected via RunPod environment variables
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"),
)

class ExtractRequest(BaseModel):
    s3_url: str          # Full S3 URI: s3://bucket-name/path/to/invoice.pdf
    prompt: Optional[str] = None

VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"

DEFAULT_PROMPT = """Extract the key information from this invoice and return it as strict JSON.
The JSON should contain fields such as invoice_number, date, total_amount, vendor_name, and line_items.
Ensure that the output is only valid JSON, without any markdown formatting or additional text."""


def download_from_s3(s3_url: str) -> tuple[bytes, str]:
    """Downloads a file from S3 and returns (raw_bytes, mime_type)."""
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError(f"Invalid S3 URL scheme: '{parsed.scheme}'. Must start with 's3://'")

    bucket = parsed.netloc
    key = parsed.path.lstrip("/")

    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        file_bytes = response["Body"].read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Failed to download from S3: {str(e)}")

    # Detect MIME type from file extension
    mime_type, _ = mimetypes.guess_type(key)
    if mime_type is None:
        mime_type = "application/octet-stream"

    return file_bytes, mime_type


def parse_vllm_response(content: str) -> dict:
    """Attempts to parse JSON from the model response, stripping any markdown wrappers."""
    clean = content.strip()
    if clean.startswith("```json"):
        clean = clean[7:]
    if clean.startswith("```"):
        clean = clean[3:]
    if clean.endswith("```"):
        clean = clean[:-3]
    return json.loads(clean.strip())


@app.post("/extract")
async def extract(request: ExtractRequest):
    # 1. Download file from S3
    file_bytes, mime_type = download_from_s3(request.s3_url)

    # 2. Encode to base64 data URI
    b64_data = base64.b64encode(file_bytes).decode("utf-8")
    data_uri = f"data:{mime_type};base64,{b64_data}"

    # 3. Build vLLM payload
    prompt_text = request.prompt if request.prompt else DEFAULT_PROMPT
    payload = {
        "model": "datalab-to/chandra-ocr-2",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_uri},
                    },
                ],
            }
        ],
        "temperature": 0.0,
    }

    # 4. Forward to vLLM
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(VLLM_ENDPOINT, json=payload)
            response.raise_for_status()
            response_data = response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach vLLM engine: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"vLLM engine error: {e.response.text}")

    # 5. Parse and return JSON output
    try:
        content = response_data["choices"][0]["message"]["content"]
        return parse_vllm_response(content)
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="vLLM returned invalid JSON structure")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Unexpected vLLM response format: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
