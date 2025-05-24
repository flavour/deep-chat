import os
import requests
from fastapi import APIRouter, HTTPException, Request, File, UploadFile, Form
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from io import BytesIO
import json # Make sure json is imported

router = APIRouter()

# Make sure to set the STABILITY_API_KEY environment variable

class StabilityTextToImageBody(BaseModel):
    text_prompts: List[Dict[str, Any]] # Structure based on Stability AI API
    # e.g., [{"text": "A lighthouse on a stormy night", "weight": 1}]
    # Add other parameters like cfg_scale, height, width, samples, steps etc.
    # Refer to Stability AI documentation for all options
    cfg_scale: Optional[float] = None
    clip_guidance_preset: Optional[str] = None
    height: Optional[int] = None
    width: Optional[int] = None
    sampler: Optional[str] = None
    samples: Optional[int] = None
    seed: Optional[int] = None
    steps: Optional[int] = None
    style_preset: Optional[str] = None


class StabilityService:
    def __init__(self):
        self.api_key = os.getenv("STABILITY_API_KEY")
        self.api_host = os.getenv("API_HOST", "https://api.stability.ai")
        if not self.api_key:
            print("STABILITY_API_KEY not set")
            # Handle missing API key

    def _get_engine_id(self, model_name: Optional[str] = "stable-diffusion-xl-1024-v1-0"):
        # This method can be expanded to select different engines based on a model_name parameter
        # For now, it defaults to a common Stable Diffusion XL model.
        # See https://platform.stability.ai/docs/features/list-engines for available engines
        return model_name

    def _handle_stability_response(self, response: requests.Response):
        if response.status_code != 200:
            try:
                error_details = response.json()
                # The error structure can vary, try to get a meaningful message
                message = error_details.get("message", response.text)
                name = error_details.get("name", "Unknown Error")
                errors = error_details.get("errors", [])
                if errors: # If there's a list of errors, append them
                    message += " (" + "; ".join(errors) + ")"
                raise HTTPException(status_code=response.status_code, detail=f"{name}: {message}")
            except ValueError: # If response is not JSON
                raise HTTPException(status_code=response.status_code, detail=response.text)
        
        # If successful, Stability AI API returns image(s) in 'artifacts'
        response_json = response.json()
        artifacts = response_json.get("artifacts")
        if not artifacts:
            raise HTTPException(status_code=500, detail="No artifacts found in Stability AI response.")
        
        # Deep Chat expects files in a specific format
        # https://deepchat.dev/docs/connect/#Response (see 'files' property)
        # We'll return base64 encoded images as per Stability AI's common response
        files_response = []
        for i, artifact in enumerate(artifacts):
            if artifact.get("base64"):
                files_response.append({
                    "type": "image", # Or "image_base64" if Deep Chat has a specific type for it
                    "src": f"data:image/png;base64,{artifact['base64']}",
                    "filename": f"image_{i}.png" # Optional filename
                })
            # Add handling for other artifact types if necessary (e.g. video, text)
        
        if not files_response:
            raise HTTPException(status_code=500, detail="No image data found in Stability AI artifacts.")
            
        return {"files": files_response}


    def text_to_image(self, body: StabilityTextToImageBody, model: Optional[str] = None):
        if not self.api_key:
            raise HTTPException(status_code=500, detail="Stability API key not configured.")
        if not body.text_prompts:
            raise HTTPException(status_code=400, detail="No text_prompts provided.")

        engine_id = self._get_engine_id(model or "stable-diffusion-xl-1024-v1-0")
        
        payload = body.dict(exclude_none=True) # Converts Pydantic model to dict, excluding None values
        
        try:
            response = requests.post(
                f"{self.api_host}/v1/generation/{engine_id}/text-to-image",
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "Authorization": f"Bearer {self.api_key}"
                },
                json=payload # Send the whole body as JSON
            )
            return self._handle_stability_response(response)
        except requests.exceptions.RequestException as e:
            print(f"Stability AI API request error (text-to-image): {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to Stability AI: {e}")
        except Exception as e:
            print(f"Stability AI service error (text-to-image): {e}")
            raise HTTPException(status_code=500, detail=str(e))


    async def image_to_image(self, request: Request, model: Optional[str] = None):
        if not self.api_key:
            raise HTTPException(status_code=500, detail="Stability API key not configured.")

        engine_id = self._get_engine_id(model or "stable-diffusion-xl-1024-v1-0") 

        form_data = await request.form()
        init_image_file = form_data.get("init_image") 
        
        if not init_image_file or not isinstance(init_image_file, UploadFile):
            raise HTTPException(status_code=400, detail="Initial image (init_image) is required as part of the form.")

        # Default text prompts if not provided or empty
        text_prompts_json = form_data.get("text_prompts")
        if not text_prompts_json: # Check if it's None or empty string
            text_prompts = [{"text": "Transform this image", "weight": 1}]
        else:
            try:
                text_prompts = json.loads(text_prompts_json)
                if not text_prompts: # Check if it's an empty list
                     text_prompts = [{"text": "Transform this image", "weight": 1}]
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for text_prompts.")
        

        files = {'init_image': (init_image_file.filename, await init_image_file.read(), init_image_file.content_type)}
        
        payload = {"text_prompts": text_prompts}
        # Populate payload with other form fields, converting types as necessary
        for key, value in form_data.items():
            if key not in ["init_image", "text_prompts"] and key not in payload:
                # Specific handling for known numeric/boolean parameters for Stability AI
                if key in ["cfg_scale", "image_strength", "step_schedule_start", "step_schedule_end"]:
                    try: payload[key] = float(value)
                    except ValueError: raise HTTPException(status_code=400, detail=f"Invalid value for {key}: must be a number.")
                elif key in ["steps", "seed", "samples"]: # samples might not be applicable for i2i for all engines
                    try: payload[key] = int(value)
                    except ValueError: raise HTTPException(status_code=400, detail=f"Invalid value for {key}: must be an integer.")
                # Add other specific parameter conversions here if needed
                else:
                    payload[key] = value # Default to string if not a known numeric/boolean
        
        try:
            response = requests.post(
                f"{self.api_host}/v1/generation/{engine_id}/image-to-image",
                headers={
                    "Accept": "application/json", 
                    "Authorization": f"Bearer {self.api_key}"
                },
                files=files,
                data=payload
            )
            return self._handle_stability_response(response)
        except requests.exceptions.RequestException as e:
            print(f"Stability AI API request error (image-to-image): {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to Stability AI: {e}")
        except Exception as e:
            print(f"Stability AI service error (image-to-image): {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if isinstance(init_image_file, UploadFile):
              await init_image_file.close()


    async def image_upscale(self, files: List[UploadFile], model: Optional[str] = "esrgan-v1-x2plus", width: Optional[int] = Form(None), height: Optional[int] = Form(None)):
        if not self.api_key:
            raise HTTPException(status_code=500, detail="Stability API key not configured.")
        if not files:
            raise HTTPException(status_code=400, detail="No image file provided for upscaling.")

        image_file = files[0] 
        engine_id = model # For upscale, the model is the engine_id, e.g., "esrgan-v1-x2plus"
        
        form_payload = {} 
        if width: form_payload['width'] = str(width) # API expects string for form data
        if height: form_payload['height'] = str(height)
        
        image_content = await image_file.read()

        try:
            response = requests.post(
                f"{self.api_host}/v1/generation/{engine_id}/image-to-image/upscale",
                headers={
                    "Accept": "application/json", 
                    "Authorization": f"Bearer {self.api_key}"
                },
                files={'image': (image_file.filename, image_content, image_file.content_type)},
                data=form_payload
            )
            return self._handle_stability_response(response)
        except requests.exceptions.RequestException as e:
            print(f"Stability AI API request error (upscale): {e}")
            raise HTTPException(status_code=500, detail=f"Error connecting to Stability AI: {e}")
        except Exception as e:
            print(f"Stability AI service error (upscale): {e}")
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            if isinstance(image_file, UploadFile):
              await image_file.close()


stability_service = StabilityService()

@router.post("/text-to-image")
async def stabilityai_text_to_image_endpoint(body: StabilityTextToImageBody, model: Optional[str] = Form(None)):
    # The 'model' parameter in Stability AI usually refers to the engine_id.
    # It can be passed via query, form, or be part of the JSON body.
    # If passed as Form, it's separate from the JSON body.
    # If model is intended to be part of the JSON, it should be in StabilityTextToImageBody.
    # For this setup, model from Form acts as an override or primary if not in JSON.
    return stability_service.text_to_image(body, model=model)

@router.post("/image-to-image")
async def stabilityai_image_to_image_endpoint(request: Request, model: Optional[str] = Form(None)):
    return await stability_service.image_to_image(request, model=model)

@router.post("/image-upscale")
async def stabilityai_image_upscale_endpoint(
    files: List[UploadFile] = File(...), 
    model: Optional[str] = Form("esrgan-v1-x2plus"), # Default upscale model
    width: Optional[int] = Form(None), # Optional: target width for upscaling
    height: Optional[int] = Form(None) # Optional: target height for upscaling
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded for upscaling")
    return await stability_service.image_upscale(files, model=model, width=width, height=height)
