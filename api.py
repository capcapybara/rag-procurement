from main import rag_chain

import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from langchain.chat_models import ChatOpenAI
from langchain.schema.runnable import RunnableLambda
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Initialize FastAPI
app = FastAPI()


# Create a Runnable for streaming (simple example)
async def stream_response(query: str):
    async for chunk in rag_chain.astream(query):
        yield str(chunk.content)


# Pydantic schema for input data
class TextRequest(BaseModel):
    text: str


# FastAPI endpoint to handle streaming requests
@app.post("/process/stream")
async def process_text_stream(request: TextRequest):
    try:
        # Use StreamingResponse to stream the result progressively
        return StreamingResponse(stream_response(request.text), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the API with: uvicorn main:app --reload
