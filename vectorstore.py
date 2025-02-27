# indexing pdf files using langchain

from langchain.retrievers import ParentDocumentRetriever
from langchain_qdrant import QdrantVectorStore, FastEmbedSparse, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
)

from langchain.storage import LocalFileStore, create_kv_docstore, InMemoryStore

from langchain_community.document_loaders import TextLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from embedder import Embeddings


print("Initing embedder")
embedder = Embeddings()

path = "./data/"

# get files in the directory


sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
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


vectorstore = QdrantVectorStore(
    client=client,
    collection_name="data",
    embedding=embedder,
    sparse_embedding=sparse_embeddings,
    sparse_vector_name="sparse",
    retrieval_mode=RetrievalMode.HYBRID,
)


local_file_store = LocalFileStore("./doc_store")
doc_store = create_kv_docstore(local_file_store)

doc_kv_fs = LocalFileStore("./doc_kv_store")
doc_kv = create_kv_docstore(doc_kv_fs)

child_splitter = RecursiveCharacterTextSplitter(chunk_size=400)
retriever = ParentDocumentRetriever(
    vectorstore=vectorstore,
    docstore=doc_store,
    child_splitter=child_splitter,
    search_kwargs={"k": 20},
)
