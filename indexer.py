# indexing pdf files using langchain
import re

from embedder import Embeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document


#  law_section_splitter split the document into sections based on keywords "มาตรา {i}"
def law_section_splitter(docs: list[Document]) -> list[Document]:
    merged = "".join([doc.page_content for doc in docs])
    sections: list[Document] = []
    i = 1
    while True:
        next_i = merged.find(f"มาตรา {i}")
        if next_i == -1:
            break
        data = merged[:next_i]
        merged = merged[next_i:]
        ref = [x for x in re.findall(r"มาตรา (\d+)", data) if int(x) != i - 1]
        ref_commasep = ",".join(ref)
        sections.append(
            Document(
                page_content=data,
                metadata={"section": str(i - 1), "ref": ref_commasep},
            )
        )
        i += 1
    ref = [x for x in re.findall(r"มาตรา (\d+)", data) if int(x) != i - 1]
    ref_commasep = ",".join(ref)
    sections.append(
        Document(
            page_content=merged,
            metadata={"section": str(i - 1), "ref": ref_commasep},
        )
    )
    return sections


def replace_thai_number(string: str) -> str:
    string = string.replace("๐", "0")
    string = string.replace("๑", "1")
    string = string.replace("๒", "2")
    string = string.replace("๓", "3")
    string = string.replace("๔", "4")
    string = string.replace("๕", "5")
    string = string.replace("๖", "6")
    string = string.replace("๗", "7")
    string = string.replace("๘", "8")
    string = string.replace("๙", "9")
    return string


file_path = "./data/T_0021.pdf"

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
    doc.page_content = doc.page_content.replace(" า", "ำ")
    doc.page_content = replace_thai_number(doc.page_content)
    docs.append(doc)

print("Initing embedder")
embedder = Embeddings()

print("Splitting text")
# text_splitter = TokenTextSplitter(
#     chunk_size=1024,
#     chunk_overlap=256,
# )
all_splits = law_section_splitter(docs)

print(all_splits)
vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=embedder, persist_directory="./chroma_db"
)
