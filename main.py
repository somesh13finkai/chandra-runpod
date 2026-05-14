import io
import os
import base64
import mimetypes
import boto3
import torch
from PIL import Image
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from transformers import AutoProcessor, AutoModelForCausalLM
from qwen_vl_utils import process_vision_info

MODEL_ID = "datalab-to/chandra-ocr-2"
model = None
processor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, processor
    print(f"Loading {MODEL_ID}...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    print("Model ready!")
    yield

app = FastAPI(title="Chandra OCR API", lifespan=lifespan)

# S3 client — credentials from RunPod env vars
s3_client = boto3.client(
    "s3",
    aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
    region_name=os.environ.get("AWS_DEFAULT_REGION", "ap-south-1"),
)

DEFAULT_PROMPT = "Extract all information from this invoice and return it as JSON."

class ExtractRequest(BaseModel):
    s3_url: str
    prompt: Optional[str] = None


def download_from_s3(s3_url: str) -> bytes:
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise HTTPException(status_code=400, detail="URL must start with s3://")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        return response["Body"].read()
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"S3 download failed: {str(e)}")


@app.post("/extract")
async def extract(request: ExtractRequest):
    # 1. Download from S3
    file_bytes = download_from_s3(request.s3_url)

    # 2. Detect MIME and encode as data URI for qwen_vl_utils
    key = urlparse(request.s3_url).path
    mime_type, _ = mimetypes.guess_type(key)
    if mime_type is None:
        mime_type = "image/jpeg"
    b64 = base64.b64encode(file_bytes).decode()
    data_uri = f"data:{mime_type};base64,{b64}"

    # 3. Build messages
    prompt_text = request.prompt or DEFAULT_PROMPT
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": data_uri},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 4. Process through model
    try:
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to("cuda")

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)

        # Strip the input tokens from output
        output_ids = generated_ids[:, inputs.input_ids.shape[1]:]
        output = processor.batch_decode(output_ids, skip_special_tokens=True)

        return {"output": output[0]}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
