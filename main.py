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

# S3 client - credentials injected via RunPod environment variables
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
MODEL_NAME = "datalab-to/chandra-ocr-1"

DEFAULT_PROMPT = "Extract all information from this invoice and return it as JSON."


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

    mime_type, _ = mimetypes.guess_type(key)
    if mime_type is None:
        mime_type = "application/octet-stream"

    return file_bytes, mime_type


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
        "model": MODEL_NAME,
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

    # 4. Forward to vLLM and return raw response as-is
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(VLLM_ENDPOINT, json=payload)
            response.raise_for_status()
            response_data = response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to reach vLLM engine: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"vLLM engine error: {e.response.text}")

    # 5. Return the raw model output — no parsing, no modification
    raw_content = response_data["choices"][0]["message"]["content"]
    return {"output": raw_content}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
