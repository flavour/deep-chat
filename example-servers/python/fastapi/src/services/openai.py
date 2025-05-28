import os
import requests
from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from io import BytesIO

router = APIRouter()

# Make sure to set the OPENAI_API_KEY environment variable

class OpenAIChatMessage(BaseModel):
    role: str
    text: str

class OpenAIChatBody(BaseModel):
    messages: List[OpenAIChatMessage]
    # Include other OpenAI specific parameters if needed, e.g., model, temperature
    model: Optional[str] = "gpt-3.5-turbo" # Default model

class OpenAIService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("OPENAI_API_KEY not set")
            # Initialization errors should ideally be handled to prevent router setup if critical

    def _get_headers(self):
        if not self.api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured.")
        return {
            "Authorization": f"Bearer {self.api_key}"
        }

    def chat(self, body: OpenAIChatBody):
        headers = self._get_headers()
        headers["Content-Type"] = "application/json"
        
        openai_body = {
            "model": body.model,
            "messages": [{"role": "assistant" if msg.role == "ai" else msg.role, "content": msg.text} for msg in body.messages]
        }
        
        try:
            response = requests.post(
                "https://api.openai.com/v1/chat/completions", json=openai_body, headers=headers)
            response.raise_for_status()
            json_response = response.json()
            if "error" in json_response:
                raise Exception(json_response["error"]["message"])

            # Sends response back to Deep Chat using the Response format:
            # https://deepchat.dev/docs/connect/#Response
            return {"text": json_response["choices"][0]["message"]["content"]}

        except requests.exceptions.RequestException as e:
            print(f"OpenAI API request error (chat): {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to OpenAI API: {e}")
        except Exception as e:
            print(f"OpenAI service error (chat): {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def image_variation(self, files: List[UploadFile]):
        if not self.api_key:
            raise HTTPException(status_code=500, detail="OpenAI API key not configured.")
        
        if not files:
            raise HTTPException(status_code=400, detail="No files provided for image variation.")

        # OpenAI API for image variations expects one image file.
        # Taking the first file if multiple are sent.
        file = files[0]
        
        # Read file content into BytesIO object as OpenAI client expects a file-like object
        file_content = await file.read()
        image_bytes_io = BytesIO(file_content)
        image_bytes_io.name = file.filename # OpenAI client library checks for .name attribute

        form_data = {'n': '1', 'size': '1024x1024'} # Example parameters, adjust as needed
        
        try:
            # Using requests to send multipart/form-data
            response = requests.post(
                "https://api.openai.com/v1/images/variations",
                headers={"Authorization": f"Bearer {self.api_key}"},
                files={'image': (file.filename, image_bytes_io, file.content_type)},
                data=form_data
            )
            response.raise_for_status()
            json_response = response.json()

            if json_response.get("data") and json_response["data"][0].get("url"):
                 # Deep Chat expects image data to be returned, typically as a URL or base64 string.
                 # The response format for files/images: https://deepchat.dev/docs/connect/#Response
                return {"files": [{"type": "image", "src": json_response["data"][0]["url"]}]}
            elif json_response.get("error"):
                raise HTTPException(status_code=response.status_code, detail=json_response["error"].get("message", "OpenAI API error"))
            else:
                raise HTTPException(status_code=500, detail="Unexpected response format from OpenAI image variation API")

        except requests.exceptions.RequestException as e:
            print(f"OpenAI API request error (image variation): {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to OpenAI API: {e}")
        except Exception as e:
            print(f"OpenAI service error (image variation): {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            image_bytes_io.close()


openai_service = OpenAIService()

@router.post("/chat")
async def openai_chat_endpoint(body: OpenAIChatBody):
    return openai_service.chat(body)

# The Flask example has /openai-chat-stream, which is not implemented here yet.
# Implementing streaming with FastAPI typically involves StreamingResponse.

@router.post("/image")
async def openai_image_endpoint(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    return await openai_service.image_variation(files)
