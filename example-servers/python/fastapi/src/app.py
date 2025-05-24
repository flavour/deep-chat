from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

# Import service routers
from services.custom import router as custom_router
from services.openai import router as openai_router
from services.huggingface import router as huggingface_router
from services.stabilityai import router as stabilityai_router
from services.cohere import router as cohere_router

# ------------------ SETUP ------------------

load_dotenv(dotenv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env'))

app = FastAPI()

# Configure CORS
# This will need to be reconfigured before taking the app to production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# ------------------ EXCEPTION HANDLERS ------------------

# Sends response back to Deep Chat using the Response format:
# https://deepchat.dev/docs/connect/#Response
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    print(f"Generic error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": str(exc)},
    )

@app.exception_handler(ConnectionError)
async def connection_error_handler(request: Request, exc: ConnectionError):
    print(f"Connection error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal service error"},
    )

# ------------------ API ROUTERS ------------------

app.include_router(custom_router, prefix="/custom")
app.include_router(openai_router, prefix="/openai")
app.include_router(huggingface_router, prefix="/huggingface")
app.include_router(stabilityai_router, prefix="/stabilityai")
app.include_router(cohere_router, prefix="/cohere")

# ------------------ START SERVER (for local development) ------------------
# This part is typically handled by Uvicorn when deploying or running locally.
# For example: uvicorn app:app --reload --port 8000
# However, including it here allows for simple `python app.py` execution for basic testing if needed.

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
