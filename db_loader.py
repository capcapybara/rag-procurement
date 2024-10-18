# indexing pdf files using langchain

import os

import jsonlines
from langchain_chroma import Chroma
from langchain_core.documents import Document

from embedder import Embeddings


def load_docs_from_jsonl(file_path) -> list[Document]:
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents


# remove ./chroma_db if it exists
if os.path.exists("./chroma_db"):
    import shutil

    print("Removing old chroma_db")
    shutil.rmtree("./chroma_db")

print("Initing embedder")
embedder = Embeddings()

path = "./data/"

# get files in the directory


all_docs = []
files = os.listdir(path)
for file in files:
    if not file.endswith(".jsonl"):
        continue
    print(file)
    docs = load_docs_from_jsonl(path + file)
    for doc in docs:
        print(doc.id)
    all_docs.extend(docs)


vectorstore = Chroma.from_documents(
    documents=all_docs, embedding=embedder, persist_directory="./chroma_db"
)
