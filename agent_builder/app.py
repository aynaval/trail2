import logging
import uuid
import time
import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, Header, Depends
from fastapi.responses import JSONResponse, StreamingResponse
from dotenv import load_dotenv

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

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Retrieve WatsonX credentials from environment variables
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
    X_IBM_THREAD_ID: Optional[str] = Header(
        None,
        alias="X-IBM-THREAD-ID",
        description="Optional header to specify the thread ID",
    ),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    logger.info(f"Received POST /chat/completions ChatCompletionRequest: {request.json()}")

    thread_id = X_IBM_THREAD_ID or ""
    if request.extra_body and request.extra_body.thread_id:
        thread_id = request.extra_body.thread_id
    logger.info(f"thread_id: {thread_id}")

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
                    message=MessageResponse(
                        role="assistant", content=all_messages[-1].content
                    ),
                    finish_reason="stop",
                )
            ],
        )
        return JSONResponse(content={"response": response.dict(), "watsonx_details": {
            "WATSONX_DEPLOYMENT_ID": WATSONX_DEPLOYMENT_ID,
            "WATSONX_API_KEY": WATSONX_API_KEY,
            "WATSONX_SPACE_ID": WATSONX_SPACE_ID,
            "WATSONX_URL": WATSONX_URL,
        }})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
