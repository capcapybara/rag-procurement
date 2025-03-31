# indexing pdf files using langchain

import os
import uuid

import jsonlines
from langchain_core.documents import Document
from langchain_qdrant import FastEmbedSparse, QdrantVectorStore, RetrievalMode
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

from embedder import Embeddings


def hash_uuid(string: str | None) -> str:
    assert string is not None
    if string == "":
        return ""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, string))


def load_docs_from_jsonl(file_path) -> list[Document]:
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            document = Document(**doc)
            document.id = hash_uuid(document.id)
            raw_ref = document.metadata["ref"]
            if raw_ref != "":
                refs = raw_ref.split(",")
                new_refs = ",".join(map(hash_uuid, refs))
                document.metadata["ref"] = new_refs
            document.metadata["ref_name"] = raw_ref
            documents.append(document)
    return documents


# remove ./chroma_db if it exists
if os.path.exists("./qdrant"):
    import shutil

    print("Removing old qdrant")
    shutil.rmtree("./qdrant")
else:
    os.makedirs("./qdrant")


if os.path.exists("./doc_store"):
    import shutil

    shutil.rmtree("./doc_store")

if os.path.exists("./doc_kv_store"):
    import shutil

    shutil.rmtree("./doc_kv_store")

# if os.path.exists("./data.ms/data.ms"):
#     import shutil

#     shutil.rmtree("./doc_kv_store")

print("Initing embedder")
embedder = Embeddings()

path = "./data/"

# get files in the directory


all_docs: list[Document] = []
files = os.listdir(path)
for file in files:
    if not file.endswith(".jsonl"):
        continue
    print(file)
    docs = load_docs_from_jsonl(path + file)
    for doc in docs:
        print(doc.metadata.get("id"))
    all_docs.extend(docs)

from vectorstore import doc_kv, meili_retriever, qdrant_retriever, retriever

# vectorstore = Chroma.from_documents(
#     documents=all_docs, embedding=embedder, persist_directory="./chroma_db"
# )

# sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
# client = QdrantClient(path="./qdrant")

# client.create_collection(
#     collection_name="data",
#     vectors_config=VectorParams(size=768, distance=Distance.COSINE),
#     sparse_vectors_config={
#         "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=True))
#     },
# )


# vector_store = QdrantVectorStore(
#     client=client,
#     collection_name="data",
#     embedding=embedder,
#     sparse_embedding=sparse_embeddings,
#     sparse_vector_name="sparse",
#     retrieval_mode=RetrievalMode.HYBRID,
# )
#

doc_kv.mset([(v.id or "", v) for v in all_docs])


meili_retriever.add_documents(all_docs)
qdrant_retriever.add_documents(all_docs, [v.id or "" for v in all_docs])
