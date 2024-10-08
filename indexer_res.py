# indexing pdf files using langchain
from langchain.text_splitter import TokenTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker

from embedder import Embeddings

file_path = "./data/all.pdf"

loader = UnstructuredPDFLoader(
    file_path,
    mode="elements",
    languages=["th"],
)
raw = loader.load()
docs = []
for doc in raw:
    if (
        doc.metadata["category"] == "Title"
        or doc.metadata["category"] == "UncategorizedText"
    ) and "มาตรา" not in doc.page_content:
        print("Skipping", doc.metadata["category"], doc.page_content[:100])
        continue
    txt = doc.page_content.replace(" า", "ำ")
    docs.append(txt)

print("Initing embedder")
embedder = Embeddings()

print("Splitting text")
text_splitter = TokenTextSplitter(
    chunk_size=1024,
    chunk_overlap=256,
)
text_splitter = SemanticChunker(embedder)
docs = text_splitter.create_documents(docs)


vectorstore = Chroma.from_documents(
    documents=docs, embedding=embedder, persist_directory="./chroma_db"
)
