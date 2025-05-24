import os
import requests
from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

router = APIRouter()

# Make sure to set the HUGGING_FACE_API_KEY environment variable

class HFChatMessage(BaseModel):
    role: str
    content: str # Deep Chat uses 'text', HuggingFace uses 'content' or specific structures

class HFConversationBody(BaseModel):
    messages: List[HFChatMessage]
    model: Optional[str] = "facebook/blenderbot-400M-distill" # Example model

class HuggingFaceService:
    def __init__(self):
        self.api_key = os.getenv("HUGGING_FACE_API_KEY")
        if not self.api_key:
            print("HUGGING_FACE_API_KEY not set")
            # Consider how to handle missing API key on startup

    def _get_headers(self):
        if not self.api_key:
            raise HTTPException(status_code=500, detail="Hugging Face API key not configured.")
        return {"Authorization": f"Bearer {self.api_key}"}

    def _query_hf_model(self, data, model_url):
        headers = self._get_headers()
        if isinstance(data, dict): # For JSON payloads
            headers["Content-Type"] = "application/json"
        # For file uploads, requests library sets Content-Type automatically for multipart/form-data
        
        try:
            response = requests.post(model_url, headers=headers, json=data if isinstance(data, dict) else None, data=data if not isinstance(data, dict) else None)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            # Model loading can take time, leading to timeouts
            # Return a specific message that Deep Chat can use to inform the user
            # https://deepchat.dev/docs/connect/#response
            # This is a simplified example; more robust handling might be needed
            print(f"HuggingFace model loading timeout for {model_url}")
            return {"error": "Model is loading, please try again in a few moments.", "model_is_loading": True}
        except requests.exceptions.RequestException as e:
            print(f"HuggingFace API request error to {model_url}: {e}")
            error_detail = f"Error connecting to HuggingFace API: {e}"
            if e.response is not None:
                try:
                    error_detail = e.response.json().get("error", error_detail)
                except ValueError: # If response is not JSON
                    error_detail = e.response.text
            raise HTTPException(status_code=500, detail=error_detail)
        except Exception as e: # Catch any other unexpected errors
            print(f"HuggingFace service error with {model_url}: {e}")
            raise HTTPException(status_code=500, detail=str(e))


    def conversation(self, body: HFConversationBody):
        model_url = f"https://api-inference.huggingface.co/models/{body.model or 'facebook/blenderbot-400M-distill'}"
        
        # Adapt Deep Chat messages to HuggingFace conversation format
        # This is a simplified adaptation; specific models might have different needs
        past_user_inputs = [msg.content for msg in body.messages if msg.role == 'user'][:-1]
        generated_responses = [msg.content for msg in body.messages if msg.role == 'ai']
        current_input = body.messages[-1].content if body.messages and body.messages[-1].role == 'user' else ""

        payload = {
            "inputs": {
                "past_user_inputs": past_user_inputs,
                "generated_responses": generated_responses,
                "text": current_input,
            }
        }
        
        json_response = self._query_hf_model(payload, model_url)
        
        if json_response.get("model_is_loading"): # Propagate model loading status
            return json_response

        if "generated_text" in json_response:
            return {"text": json_response["generated_text"]}
        elif "error" in json_response:
            # Handle cases where HF returns an error in the JSON body but with a 200 status
            raise HTTPException(status_code=500, detail=json_response["error"])
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format from HuggingFace conversation API")

    async def image_classification(self, files: List[UploadFile], model: str = "google/vit-base-patch16-224"):
        model_url = f"https://api-inference.huggingface.co/models/{model}"
        if not files:
            raise HTTPException(status_code=400, detail="No file provided for image classification.")
        
        file = files[0] # Process first file
        file_content = await file.read()
        
        json_response = self._query_hf_model(file_content, model_url)

        if json_response.get("model_is_loading"):
            return json_response

        # Example: returning raw response, this might need structuring for Deep Chat
        # Deep Chat expects a specific format for different types of data.
        # For text: {"text": "..."}
        # For files/images: {"files": [{"type": "image", "src": "url_or_base64"}]}
        # The response here is a list of classifications. You might need to format this.
        # This example returns the top classification as text.
        if isinstance(json_response, list) and json_response:
             # Assuming the response is a list of dicts with 'label' and 'score'
            top_prediction = json_response[0]
            return {"text": f"Label: {top_prediction.get('label')}, Score: {top_prediction.get('score'):.2f}"}
        elif "error" in json_response:
            raise HTTPException(status_code=500, detail=json_response["error"])
        else:
            print(f"HF image_classification unexpected response: {json_response}")
            raise HTTPException(status_code=500, detail="Unexpected response format from HuggingFace image classification API")


    async def speech_recognition(self, files: List[UploadFile], model: str = "facebook/wav2vec2-base-960h"):
        model_url = f"https://api-inference.huggingface.co/models/{model}"
        if not files:
            raise HTTPException(status_code=400, detail="No file provided for speech recognition.")

        file = files[0] # Process first file
        file_content = await file.read()

        json_response = self._query_hf_model(file_content, model_url)

        if json_response.get("model_is_loading"):
            return json_response
        
        if "text" in json_response:
            return {"text": json_response["text"]}
        elif "error" in json_response:
            raise HTTPException(status_code=500, detail=json_response["error"])
        else:
            raise HTTPException(status_code=500, detail="Unexpected response format from HuggingFace speech recognition API")


huggingface_service = HuggingFaceService()

@router.post("/conversation")
async def hf_conversation_endpoint(body: HFConversationBody):
    return huggingface_service.conversation(body)

@router.post("/image") # Matches Flask example's /huggingface-image
async def hf_image_classification_endpoint(
    files: List[UploadFile] = File(...), 
    model: Optional[str] = Form("google/vit-base-patch16-224") # Example model, can be passed as form data
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    return await huggingface_service.image_classification(files, model)

@router.post("/speech") # Matches Flask example's /huggingface-speech
async def hf_speech_recognition_endpoint(
    files: List[UploadFile] = File(...),
    model: Optional[str] = Form("facebook/wav2vec2-base-960h") # Example model
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")
    return await huggingface_service.speech_recognition(files, model)
