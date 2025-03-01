from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, SecretStr
from typing import List, Optional, Dict, Any, Literal, AsyncGenerator
import time
import uvicorn
from contextlib import asynccontextmanager
from os import getenv
from datetime import datetime
from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, SparseVectorParams, SparseIndexParams
from langchain.storage import LocalFileStore, create_kv_docstore
from langchain.retrievers import ParentDocumentRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedder import Embeddings
import json

load_dotenv()

def env(key: str) -> str:
    data = getenv(key)
    assert data is not None
    return data

# Global variables to store initialized components
retriever = None
doc_kv = None
rag_chain = None

year_th = datetime.now().year + 543

# Initialize prompts
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an AI assistant that uses a Chain of Thought (CoT) approach with reflection to answer queries about the Public Procurement and Asset Management Act for government. Follow these steps:

        1. Think through the problem **step by step** within the <thinking> tags, your thinking must have a referenceable data from context, focus on what could be done, what couldn't, what's the edge case, are there any other way.
        2. Reflect on your thinking to check for any errors or improvements within the <reflection> tags.
        3. Make any necessary adjustments based on your reflection, repeat the process until you have satisfying answer.
        4. Provide your final, concise answer within the <output> tags with the reference to the context and data you get from.

        Important: The <thinking> and <reflection> sections are for your internal reasoning process only. 
        Do not include any part of the final answer in these sections. 
        The actual response to the query must be entirely contained within the <output> tags.

        Use the following format for your response:
        <thinking>
            [Your step-by-step reasoning goes here. This is your internal thought process, not the final answer.]
                    <reflection>
               [Your reflection on your reasoning, checking for errors or improvements]
            </reflection>
        
            [Any adjustments to your thinking based on your reflection]
            <reflection>
                [Another reflection if needed]
            </reflection>
        </thinking>

        
        <output>
        [Your final, concise answer to the query with source of the information. This is the only part that will be shown to the user.]
        </output>

Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.

Use the following pieces of retrieved context as a evidence and reference to your reasoning and answering. You must convert any Thai numerals or thai word that mean the number to Arabic numerals. You can ask for more explanation to get better understanding. And please answer with politeness manner for Thai people with male-based pronouns (ครับ).

    + Before you answer the question, please read the context below carefully. If you need more information, you can ask for more explanation.
    + Context contents that are inside "[]" means that it is a data that later added to be more informative to you the AI, which is not a part of the original content. **You must use it to answer the question**.
    + This year is พ.ศ. {year_th}, data or question that related to time might get changed due to postponded or delayed events. Please use the current year as a reference only if needed.
    + You must always provided all sources of the answer (label as "Label/Origin") at the end, give it in specific name that you can refer to the source later.
  
""",
        ),
        ("system", "Context: {context}"),
        ("human", "{question}"),
    ]
)

rewrite_prompt = ChatPromptTemplate.from_template(
    """You are an AI language model assistant. Your task is
    to generate sentences of important keywords or phrases that is relevant to the user question to retrieve relevant Thai legal documents like act of legislation from a vector database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    sentence. separated by newlines without any kind of suffix (number, dash, etc).

    You should generate using these strategies, one for each if not specified:
        1. Step back and paraphrase a question to a more generic step-back question, which is easier to answer.
        2. Provide a question that is more specific or detailed than the original question.
        3. Paraphrase the question to be easier to find.
        4. Paraphrase the question in opposite manner, e.g. if the question is asking "Is it an A", try paraphrase is to "Is it not A".
        5. If the sentence is composed of multiple parts, provide a question that focuses on one of those parts. If the sentence has only one part, ignore this strategy.

    Known opposite words:
    - เสียภาษี : ยกเว้นภาษี

    And this year is พ.ศ. {year_th}, data or question that related to time might get changed due to postponded or delayed events. Please use the current year as a reference, do not use any relative time question at all.

    Original question: {question}"""
)

def split_query(text: str):
    raw = [v.strip() for v in text.split("\n")]
    res = []
    for v in raw:
        if v:
            print("\t- split_query", v)
            res.append(v)
    return res

def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        loads(doc)
        for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results[:20]

llm = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global retriever, doc_kv, rag_chain, year_th, llm
    
    print("Initializing Qdrant and other components...")
    
    # Initialize LLMs
    llm = ChatOpenAI(
        api_key=SecretStr(env("OPENROUTER_API_KEY")),
        base_url=env("OPENROUTER_BASE_URL"),
        model="deepseek/deepseek-r1-distill-qwen-32b",
        temperature=0,
        timeout=None,
        max_retries=10,
    )
    
    fast_llm = ChatOpenAI(
        api_key=SecretStr(env("OPENROUTER_API_KEY")),
        base_url=env("OPENROUTER_BASE_URL"),
        model="qwen/qwen-turbo",
        temperature=0,
        timeout=None,
        max_retries=10,
    )
    
    # Initialize embeddings
    embedder = Embeddings()
    sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
    
    # Initialize Qdrant client
    client = QdrantClient(path="./qdrant")
    
    try:
        client.create_collection(
            collection_name="data",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            sparse_vectors_config={
                "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
            },
        )
    except Exception as e:
        print("Collection might already exist:", e)
    
    # Initialize vector store
    vectorstore = QdrantVectorStore(
        client=client,
        collection_name="data",
        embedding=embedder,
        sparse_embedding=sparse_embeddings,
        sparse_vector_name="sparse",
        retrieval_mode=RetrievalMode.HYBRID,
    )
    
    # Initialize document stores
    local_file_store = LocalFileStore("./doc_store")
    doc_store = create_kv_docstore(local_file_store)
    
    doc_kv_fs = LocalFileStore("./doc_kv_store")
    doc_kv = create_kv_docstore(doc_kv_fs)
    
    # Initialize retriever
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=doc_store,
        child_splitter=child_splitter,
        search_kwargs={"k": 20},
    )
    
    # Initialize RAG chain components
    generate_query = rewrite_prompt | fast_llm | StrOutputParser() | split_query
    multi_query_retriever = generate_query | retriever.map() | reciprocal_rank_fusion
    

    rag_chain = (
        {
            "year_th": lambda _: year_th,
            "context": multi_query_retriever | get_related_docs | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("Initialization complete!")
    yield
    # Cleanup (if needed)

app = FastAPI(lifespan=lifespan)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0
    stream: Optional[bool] = False

class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str

class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"]
    created: int
    model: str
    choices: List[Choice]
    usage: Dict[str, int]

class Model(BaseModel):
    id: str
    object: Literal["model"]
    created: int
    owned_by: str

class ModelList(BaseModel):
    object: Literal["list"]
    data: List[Model]

AVAILABLE_MODELS = [
    {
        "id": "rag-procurement",
        "object": "model",
        "created": 1706745600,  # February 1, 2024
        "owned_by": "procurement"
    }
]

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    """List available models with OpenAI-compatible format."""
    return ModelList(
        object="list",
        data=[Model(**model) for model in AVAILABLE_MODELS]
    )

def get_related_docs(docs):
    result = []
    refs = []
    for doc in docs:
        result.append(doc)
        ref = doc.metadata.get("ref")
        if ref:
            refs.extend(ref.split(","))
    
    recursive_query(refs, set(), result)
    return result

def recursive_query(ref: list[str], done: set[str], result):
    if not ref:
        return

    related_docs = doc_kv.mget(ref)

    if not related_docs:
        return
    new_ref = []
    for i, doc in enumerate(related_docs):
        if doc is None:
            print("DOC NOT FOUND!!!", ref[i])
            continue
        result.append(doc)
        ref_doc = doc.metadata.get("ref")
        if ref_doc:
            new_ref.extend(ref_doc.split(","))
    done.update(ref)
    new_ref = list(set(new_ref).difference(done))
    recursive_query(new_ref, done, result)

def format_docs(docs):
    datas = []
    for doc in docs:
        print(doc.metadata.get("id"))
        data = f"""
Label/Origin: {doc.metadata.get("id")}
Content: {doc.page_content}
"""
        datas.append(data)
    res = "\n---\n\n".join(datas)
    print(f"Estimated total length: {len(res)/5}")
    return res

class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"]
    created: int
    model: str
    choices: List[Dict[str, Any]]

async def stream_response(response_stream: AsyncGenerator) -> AsyncGenerator[str, None]:
    async for chunk in response_stream:
        response = ChatCompletionChunk(
            id=str(round(time.time())),
            object="chat.completion.chunk",
            created=round(time.time()),
            model="rag-procurement",
            choices=[{
                "index": 0,
                "delta": {"role": "assistant", "content": chunk},
                "finish_reason": None
            }]
        )
        yield f"data: {json.dumps(response.dict())}\n\n"
    
    # Send the final chunk with finish_reason: "stop"
    final_response = ChatCompletionChunk(
        id="chatcmpl-" + str(round(time.time())),
        object="chat.completion.chunk",
        created=round(time.time()),
        model="rag-procurement",
        choices=[{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    )
    yield f"data: {json.dumps(final_response.dict())}\n\n"
    yield "data: [DONE]\n\n"

@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest):
    global year_th, llm
    try:
        # Extract the last user message
        last_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            None
        )
        
        if "Create a concise, 3-5 word title" in last_message or "Generate 1-3 broad tags" in last_message:
            # Use fast_llm directly for these simple generation tasks
            response = await llm.ainvoke(
                [
                    {"role": "system", "content": "You are a helpful assistant that generates concise titles and tags."},
                    {"role": "user", "content": last_message}
                ]
            )
            return ChatCompletionResponse(
                id="chatcmpl-" + str(round(time.time())),
                object="chat.completion",
                created=round(time.time()),
                model=request.model,
                choices=[
                    Choice(
                        index=0,
                        message=Message(role="assistant", content=response.content),
                        finish_reason="stop"
                    )
                ],
                usage={
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            )
        
        if not last_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Handle streaming
        if request.stream:
            response_stream = rag_chain.astream({"question": last_message, "year_th": year_th})
            return StreamingResponse(
                stream_response(response_stream),
                media_type="text/event-stream"
            )

        # Non-streaming response
        response = await rag_chain.ainvoke({"question": last_message, "year_th": year_th})
        
        return ChatCompletionResponse(
            id="chatcmpl-" + str(round(time.time())),
            object="chat.completion",
            created=round(time.time()),
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response),
                    finish_reason="stop"
                )
            ],
            usage={
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0
            }
        )
    except Exception as e:
        print(f"Error in chat completion: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True) 