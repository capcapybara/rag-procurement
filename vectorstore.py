# indexing pdf files using langchain

from dotenv import load_dotenv
from langchain.retrievers import EnsembleRetriever, ParentDocumentRetriever
from langchain.storage import InMemoryStore, LocalFileStore, create_kv_docstore
from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from langchain_text_splitters import RecursiveCharacterTextSplitter
from meilisearch import models
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from custom_meili import Meilisearch

load_dotenv()

from embedder import Embeddings

print("Initing embedder")
embedder = Embeddings()
# sparse_embeddings = SparseEmbeddings()

path = "./data/"


# get files in the directory


# sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
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
    print(
        "Error while creating collection, can be ignore if it's already created collection error:",
        e,
    )

qdrant_vectorstore = QdrantVectorStore(
    client=client,
    collection_name="data",
    embedding=embedder,
    retrieval_mode=RetrievalMode.DENSE,
)

embedders = {"default": {"source": "userProvided", "dimensions": 768}}

meili_vectorstore = Meilisearch(
    embedding=embedder,
    embedders=embedders,
    url="http://localhost:7700",
    api_key="a934cc544543815c4f7c0fed655daee00e8f3235b32c91ac0f228e7408e9c699",
)

local_file_store = LocalFileStore("./doc_store")
doc_store = create_kv_docstore(local_file_store)

doc_kv_fs = LocalFileStore("./doc_kv_store")
doc_kv = create_kv_docstore(doc_kv_fs)


child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
qdrant_retriever = ParentDocumentRetriever(
    vectorstore=qdrant_vectorstore,
    docstore=doc_store,
    child_splitter=child_splitter,
    search_kwargs={"k": 20},
)

meili_retriever = meili_vectorstore.as_retriever()

retriever = EnsembleRetriever(
    retrievers=[qdrant_retriever, meili_retriever], weights=[0.5, 0.5]
)
