# indexing pdf files using langchain
import os
import re
import typing as t

import jsonlines
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_core.documents import Document

prefix = {
    "พรบ2562.section.pdf": "พระราชบัญญัติ ภาษีที่ดินและสิ่งปลูกสร้าง พ.ศ. 2562",
    "พรบ-จัดซื้อจัดจ้าง-2560.section.pdf": "พระราชบัญญัติการจัดซื้อจัดจ้างและการบริหารพัสดุภาครัฐ พ.ศ. ๒๕๖๐",
}


def get_prefix(file: str) -> str:
    if file in prefix:
        return prefix[file] + " - "
    return "unknown - "


#  law_section_splitter split the document into sections based on keywords "มาตรา {i}"
def law_section_splitter(prefix: str, docs: list[Document]) -> list[Document]:
    merged = "".join([doc.page_content for doc in docs])
    sections: list[Document] = []
    i = 1
    while True:
        next_i = merged.find(f"มาตรา {i}")
        if next_i == -1:
            break
        data = merged[:next_i]
        merged = merged[next_i:]
        ref = [
            prefix + "มาตรา " + x
            for x in re.findall(r"มาตรา (\d+)", data)
            if int(x) != i - 1
        ]
        ref_commasep = ",".join(ref)
        id = prefix + "มาตรา " + str(i - 1)
        sections.append(
            Document(
                id=id,
                page_content=data,
                metadata={
                    "id": id,
                    "ref": ref_commasep,
                },
            )
        )
        i += 1
    ref = [
        prefix + "มาตรา " + x
        for x in re.findall(r"มาตรา (\d+)", data)
        if int(x) != i - 1
    ]
    ref_commasep = ",".join(ref)
    id = prefix + "มาตรา " + str(i - 1)
    sections.append(
        Document(
            id=id,
            page_content=merged,
            metadata={"id": id, "ref": ref_commasep},
        )
    )
    return sections


def load_docs_from_jsonl(file_path) -> list[Document]:
    documents = []
    with jsonlines.open(file_path, mode="r") as reader:
        for doc in reader:
            documents.append(Document(**doc))
    return documents


def save_docs_to_jsonl(documents: t.Iterable[Document], file_path: str) -> None:
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            writer.write(doc.dict())


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


path = "./raw/"

# get files in the directory
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


files = os.listdir(path)

for file in files:
    if not file.endswith(".section.pdf"):
        continue
    file_path = os.path.join(path, file)

    loader = UnstructuredPDFLoader(
        file_path,
        # mode="elements",
        languages=["th"],
    )
    raw = loader.load()
    docs = []
    for doc in raw:
        # if (
        #     (
        #         doc.metadata["category"] == "Title"
        #         or doc.metadata["category"] == "UncategorizedText"
        #     )
        #     and "มาตรา" not in doc.page_content
        #     and ("เล่ม " in doc.page_content or "หน้า " in doc.page_content)
        # ):

        #     print("Skipping", doc.metadata["category"], doc.page_content[:100])
        #     continue
        doc.page_content = doc.page_content.replace(" า", "ำ")
        doc.page_content = replace_thai_number(doc.page_content)
        docs.append(doc)

    print(f"Splitting text from {file}")

    prefix_str = get_prefix(file)
    all_splits = law_section_splitter(prefix_str, docs)

    save_docs_to_jsonl(all_splits, "./data/" + file.replace(".pdf", ".jsonl"))
