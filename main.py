import json
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Serverless GPU API")

class ExtractRequest(BaseModel):
    base64_image: str
    prompt: Optional[str] = None

VLLM_ENDPOINT = "http://localhost:8000/v1/chat/completions"

DEFAULT_PROMPT = """Extract the key information from this invoice and return it as strict JSON.
The JSON should contain fields such as invoice_number, date, total_amount, vendor_name, and line_items.
Ensure that the output is only valid JSON, without any markdown formatting or additional text."""

def parse_vllm_response(content: str) -> dict:
    """Attempts to parse JSON from the model response, stripping any markdown wrappers."""
    clean_content = content.strip()
    if clean_content.startswith("```json"):
        clean_content = clean_content[7:]
    if clean_content.startswith("```"):
        clean_content = clean_content[3:]
    if clean_content.endswith("```"):
        clean_content = clean_content[:-3]
    return json.loads(clean_content.strip())

@app.post("/extract")
async def extract(request: ExtractRequest):
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
                        "image_url": {
                            "url": request.base64_image if request.base64_image.startswith("data:") else f"data:image/jpeg;base64,{request.base64_image}"
                        }
                    }
                ]
            }
        ],
        "temperature": 0.0,
    }

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(VLLM_ENDPOINT, json=payload)
            response.raise_for_status()
            response_data = response.json()
    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Failed to communicate with vLLM engine: {str(e)}")
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"vLLM engine returned an error: {e.response.text}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error communicating with vLLM: {str(e)}")

    try:
        content = response_data["choices"][0]["message"]["content"]
        parsed_json = parse_vllm_response(content)
        return parsed_json
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="vLLM returned invalid JSON structure")
    except (KeyError, IndexError) as e:
        raise HTTPException(status_code=500, detail=f"Unexpected response format from vLLM: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
