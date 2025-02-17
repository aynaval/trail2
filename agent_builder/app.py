import logging
import traceback
import uuid
import time
import os
from typing import Optional, Dict, Any

from fastapi import FastAPI, Header, Depends, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv
from uvicorn import run

# Your imports
from models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    MessageResponse,
)
from security import get_current_user
from utils import get_llm_sync, get_llm_stream

# Load environment variables
load_dotenv()

# -----------------------------------------------------------
# 1) Configure logging (optional, but we'll keep default)
# -----------------------------------------------------------
# By default, Uvicorn prints logs to the console. 
# So we do not override the 'log_config' or remove handlers. 
# We'll just use Python's standard logger as well.

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # or DEBUG, etc.

# If you want a simple console handler with a custom format:
console_handler = logging.StreamHandler()
console_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
console_handler.setFormatter(console_format)
logger.addHandler(console_handler)

# -----------------------------------------------------------
# 2) Create the FastAPI app and routes
# -----------------------------------------------------------
WATSONX_DEPLOYMENT_ID = os.getenv("WATSONX_DEPLOYMENT_ID")
WATSONX_API_KEY = os.getenv("WATSONX_API_KEY")
WATSONX_SPACE_ID = os.getenv("WATSONX_SPACE_ID")
WATSONX_URL = os.getenv("WATSONX_URL", "https://us-south.ml.cloud.ibm.com")

app = FastAPI()

@app.get("/")
async def home():
    return {
        "WATSONX_DEPLOYMENT_ID": WATSONX_DEPLOYMENT_ID,
        "WATSONX_API_KEY": WATSONX_API_KEY,
        "WATSONX_SPACE_ID": WATSONX_SPACE_ID,
        "WATSONX_URL": WATSONX_URL,
    }

@app.post("/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    X_IBM_THREAD_ID: Optional[str] = Header(None, alias="X-IBM-THREAD-ID"),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    logger.info(f"Received POST /chat/completions ChatCompletionRequest: {request.json()}")

    thread_id = X_IBM_THREAD_ID or ""
    if request.extra_body and request.extra_body.thread_id:
        thread_id = request.extra_body.thread_id
    logger.info(f"thread_id: {thread_id}")

    # Uncomment this line to test an error:
    # raise ValueError("Something went wrong inside /chat/completions")

    if request.stream:
        return StreamingResponse(
            get_llm_stream(request.messages, thread_id), media_type="text/event-stream"
        )
    else:
        all_messages = get_llm_sync(request.messages)
        response = ChatCompletionResponse(
            id=str(uuid.uuid4()),
            object="chat.completion",
            created=int(time.time()),
            model="wx.ai AI service",
            choices=[
                Choice(
                    index=0,
                    message=MessageResponse(role="assistant", content=all_messages[-1].content),
                    finish_reason="stop",
                )
            ],
        )
        return JSONResponse(
            content={
                "response": response.dict(),
                "watsonx_details": {
                    "WATSONX_DEPLOYMENT_ID": WATSONX_DEPLOYMENT_ID,
                    "WATSONX_API_KEY": WATSONX_API_KEY,
                    "WATSONX_SPACE_ID": WATSONX_SPACE_ID,
                    "WATSONX_URL": WATSONX_URL,
                },
            }
        )

# -----------------------------------------------------------
# 3) Global exception handler that logs to console 
#    *and* returns errors in JSON
# -----------------------------------------------------------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """
    Catch all unhandled exceptions:
    1. Log to the console (so you see it in the terminal).
    2. Return a JSON response (so you see it in Swagger/UI).
    """
    logger.error("Unhandled exception occurred!", exc_info=exc)

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "detail": "An internal server error occurred.",
            "error": str(exc),
            "traceback": traceback.format_exc(),
        },
    )

# -----------------------------------------------------------
# 4) Run with default Uvicorn logging 
#    (so errors go to console)
# -----------------------------------------------------------
if __name__ == "__main__":
    run("main:app", host="0.0.0.0", port=8000, reload=True)
