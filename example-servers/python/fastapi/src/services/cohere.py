import requests
import os
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional

router = APIRouter()

# Make sure to set the COHERE_API_KEY environment variable

class CohereChatMessage(BaseModel):
    role: str
    text: str

class CohereChatBody(BaseModel):
    messages: List[CohereChatMessage]

class CohereGenerateBody(BaseModel):
    messages: List[CohereChatMessage] # Assuming the first message contains the prompt

class CohereSummarizeBody(BaseModel):
    messages: List[CohereChatMessage] # Assuming the first message contains the text to summarize

class CohereService:
    def __init__(self):
        self.api_key = os.getenv("COHERE_API_KEY")
        if not self.api_key:
            # This will be caught by the generic exception handler in app.py
            # and return a 500 error with a message.
            # Consider if a more specific startup error or check is needed.
            print("COHERE_API_KEY not set")
            # raise HTTPException(status_code=500, detail="COHERE_API_KEY not configured")


    def _get_headers(self):
        if not self.api_key:
            raise HTTPException(status_code=500, detail="Cohere API key not configured.")
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def chat(self, body: CohereChatBody):
        headers = self._get_headers()
        chat_body_cohere_format = self.create_chat_body(body)
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/chat", json=chat_body_cohere_format, headers=headers)
            response.raise_for_status()  # Raises an HTTPError for bad responses (4XX or 5XX)
            json_response = response.json()
            if "message" in json_response and response.status_code != 200: # Cohere might return 200 with a message field for errors
                 raise HTTPException(status_code=response.status_code, detail=json_response["message"])
            if "text" not in json_response:
                 raise HTTPException(status_code=500, detail="Unexpected response format from Cohere chat API")
            return {"text": json_response["text"]}
        except requests.exceptions.RequestException as e:
            print(f"Cohere API request error (chat): {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to Cohere API: {e}")
        except Exception as e:
            print(f"Cohere service error (chat): {e}")
            # Catching other exceptions to ensure a structured error response
            raise HTTPException(status_code=500, detail=str(e))


    @staticmethod
    def create_chat_body(body: CohereChatBody):
        if not body.messages:
            raise HTTPException(status_code=400, detail="No messages provided")
        return {
            'query': body.messages[-1].text,
            'chat_history': [
                {
                    'user_name': 'CHATBOT' if message.role == 'ai' else 'USER',
                    'text': message.text
                } for message in body.messages[:-1]
            ],
        }

    def generate_text(self, body: CohereGenerateBody):
        headers = self._get_headers()
        if not body.messages or not body.messages[0].text:
            raise HTTPException(status_code=400, detail="No prompt text provided in messages")
        generation_body = {"prompt": body.messages[0].text}
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/generate", json=generation_body, headers=headers)
            response.raise_for_status()
            json_response = response.json()
            if "message" in json_response and response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=json_response["message"])
            if not json_response.get("generations"):
                 raise HTTPException(status_code=500, detail="Unexpected response format from Cohere generate API")
            return {"text": json_response["generations"][0]["text"]}
        except requests.exceptions.RequestException as e:
            print(f"Cohere API request error (generate): {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to Cohere API: {e}")
        except Exception as e:
            print(f"Cohere service error (generate): {e}")
            raise HTTPException(status_code=500, detail=str(e))

    def summarize_text(self, body: CohereSummarizeBody):
        headers = self._get_headers()
        if not body.messages or not body.messages[0].text:
            raise HTTPException(status_code=400, detail="No text provided in messages for summarization")
        summarization_body = {"text": body.messages[0].text}
        try:
            response = requests.post(
                "https://api.cohere.ai/v1/summarize", json=summarization_body, headers=headers)
            response.raise_for_status()
            json_response = response.json()
            if "message" in json_response and response.status_code != 200:
                raise HTTPException(status_code=response.status_code, detail=json_response["message"])
            if "summary" not in json_response:
                 raise HTTPException(status_code=500, detail="Unexpected response format from Cohere summarize API")
            return {"text": json_response["summary"]}
        except requests.exceptions.RequestException as e:
            print(f"Cohere API request error (summarize): {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to Cohere API: {e}")
        except Exception as e:
            print(f"Cohere service error (summarize): {e}")
            raise HTTPException(status_code=500, detail=str(e))

cohere_service = CohereService()

@router.post("/chat")
async def cohere_chat_endpoint(body: CohereChatBody):
    return cohere_service.chat(body)

@router.post("/generate")
async def cohere_generate_endpoint(body: CohereGenerateBody):
    return cohere_service.generate_text(body)

@router.post("/summarize")
async def cohere_summarize_endpoint(body: CohereSummarizeBody):
    return cohere_service.summarize_text(body)
