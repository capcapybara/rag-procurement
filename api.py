from datetime import time
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import BaseMessage
from pydantic import BaseModel
from qdrant_client.grpc.points_pb2 import json__with__int__pb2
from main import rag_chain
from time import time
import json
import uuid

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

# Define the request structure based on OpenAI's chat API
class Message(BaseModel):
    role: str  # "system", "user", or "assistant"
    content: str


class ChatRequest(BaseModel):
    model: str
    messages: list[Message]  # Initialize FastAPI


human_message_template = HumanMessagePromptTemplate.from_template("{message}")
chat_prompt = ChatPromptTemplate.from_messages([human_message_template])

app = FastAPI()


# Define the OpenAI API-like endpoint for streaming
@app.post("/v1/chat/completions")
async def completions(request: ChatRequest):
    system_message = ""
    user_message = ""

    for msg in request.messages:
        if msg.role == "system":
            system_message = msg.content
        elif msg.role == "user":
            user_message = msg.content

    # Prepare the prompt with the system message and user message
    prompt = chat_prompt.format_messages(message=user_message)

    async def generate_response(prompt: list[BaseMessage]):
        # We call the LLM chain with the prompt, token by token
        response = rag_chain.astream(prompt)
        async for chunk in response:
            print(chunk, end="", flush=True)
            yield "data: " + json.dumps(
                {
                    "id": str(uuid.uuid4()),  # You could generate a unique ID here
                    "object": "chat.completion.chunk",
                    "created": int(time()),
                    # You could generate the current timestamp
                    "model": request.model,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"role": "assistant", "content": chunk},
                            "finish_reason": None,
                        }
                    ],
                }
            ) + "\n\n"

        yield "data: " + json.dumps(
            {
                "id": str(uuid.uuid4()),  # You could generate a unique ID here
                "object": "chat.completion.chunk",
                "created": int(time()),
                # You could generate the current timestamp
                "model": request.model,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": ""},
                        "finish_reason": "stop",
                    }
                ],
            }
        ) + "\n\n"

    # body = await request.json()

    # prompt = body.get("prompt", "")

    # # Streaming response generator to yield tokens
    # async def generate_response(prompt: str):
    #     # We call the LLM chain with the prompt, token by token
    #     response = rag_chain.astream(prompt)
    #     async for chunk in response:
    #         yield chunk

    # # Return the StreamingResponse, mimicking the OpenAI streaming behavior
    return StreamingResponse(generate_response(prompt), media_type="text/event-stream")


@app.get("/v1/models")
async def models():
    return {
        "object": "list",
        "data": [
            {
                "id": "dev",
                "object": "model",
                "created": 1686935002,
                "owned_by": "dev",
            },
        ],
    }
