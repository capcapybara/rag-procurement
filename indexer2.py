# indexing pdf files using langchain
import json
import re

from langchain_chroma import Chroma
from langchain_core.documents import Document

from embedder import Embeddings


class Section:
    def __init__(self, content: str, section: int, references: list[int]):
        self.content = content
        self.references = references
        self.section = section


#  law_section_splitter split the document into sections based on keywords "มาตรา {i}"
def law_section_splitter(docs: list[Document]) -> list[Document]:
    merged = "".join([doc.page_content for doc in docs])
    sections: list[Section] = []
    i = 1
    merged = replace_thai_number(merged)
    while True:
        next_i = merged.find(f"มาตรา {i}")
        if next_i == -1:
            break

        data = merged[:next_i]
        merged = merged[next_i:]
        i += 1
        if i == 2:
            continue
        # get all x using pattern "มาตรา {x}" by regex
        ref = [int(x) for x in re.findall(r"มาตรา (\d+)", data)]
        sections.append(Section(content=data, section=i - 2, references=ref))
    ref = [int(x) for x in re.findall(r"มาตรา (\d+)", data)]
    sections.append(Section(content=merged, section=i - 2, references=ref))

    docs = section_grouping(sections)
    return docs


def dfs(g: set[(int, int)], visited: set[int], current: int, group: list[int]):
    if current in visited:
        return
    visited.add(current)
    group.append(current)
    for a, b in g:
        if a == current:
            dfs(g, visited, b, group)
        if b == current:
            dfs(g, visited, a, group)


def section_grouping(sections: list[Section]) -> list[Document]:
    docs: list[Document] = []
    # for section in sections:
    #     ref_data = []
    #     for ref in section.references:
    #         if ref == section.section:
    #             continue
    #         ref_data.append(sections[ref - 1].content)
    #     content = (section.content + "\n\n" + "\n\n".join(ref_data)).strip()
    #     doc = Document(
    #         page_content=content,
    #         metadata={
    #             "section": f"มาตรา {section.section}",
    #             "from": "พระราชบัญญัติ ภาษีที่ดินและสิ่งปลูกสร้าง พ.ศ. 2562",
    #         },
    #     )
    #     docs.append(doc)
    graph: set[(int, int)] = set()
    for section in sections:
        for ref in section.references:
            a, b = section.section, ref
            if a > b:
                a, b = b, a
            graph.add((a, b))
    visited = set()
    for a, b in graph:
        if a in visited or b in visited:
            continue
        group = []
        dfs(graph, visited, a, group)
        print(group)
        content = "\n\n".join([sections[i - 1].content for i in group])
        doc = Document(
            page_content=content,
            metadata={
                "from": "พระราชบัญญัติ ภาษีที่ดินและสิ่งปลูกสร้าง พ.ศ. 2562",
            },
        )
        docs.append(doc)
    return docs


def to_thai_number(num):
    thai_num = {
        "0": "๐",
        "1": "๑",
        "2": "๒",
        "3": "๓",
        "4": "๔",
        "5": "๕",
        "6": "๖",
        "7": "๗",
        "8": "๘",
        "9": "๙",
    }
    return "".join([thai_num[c] for c in str(num)])


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


def load_docs_from_jsonl(file_path) -> list[Document]:
    array = []
    with open(file_path, "r") as jsonl_file:
        for line in jsonl_file:
            data = json.loads(line)
            obj = Document(**data)
            array.append(obj)
    return array


print("Initing embedder")
embedder = Embeddings()

# print("Splitting text")
# # text_splitter = TokenTextSplitter(
# #     chunk_size=1024,
# #     chunk_overlap=256,
# # )
docs = load_docs_from_jsonl("data.json")
all_splits = law_section_splitter(docs)


out = "\n\n---\n\n".join([doc.page_content for doc in all_splits])
with open("output.txt", "w") as f:
    f.write(out)
vectorstore = Chroma.from_documents(
    documents=all_splits, embedding=embedder, persist_directory="./chroma_db"
)
