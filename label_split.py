# indexing pdf files using langchain
import csv
import io
import json
import os
import re
import typing as t

import jsonlines
from langchain_core.documents import Document


#  law_section_splitter split the document into sections based on keywords "มาตรา {i}"
def law_section_splitter(label: str, prefix: str, raw: str) -> list[Document]:
    print("run law section splitter")
    label = label.strip() + ": "
    prefix = prefix.strip()
    sections: list[Document] = []
    i = 1
    while True:
        next_i = raw.find(f"{prefix} {i}")
        if next_i == -1:
            print("not found", f"{prefix} {i}")
            break
        data = raw[:next_i]
        print()
        print(next_i, data)
        raw = raw[next_i:]
        ref = [
            label + prefix + " " + x
            for x in re.findall(rf"{prefix}\s+(\d+)", data)
            if int(x) != i - 1
        ]
        ref_commasep = ",".join(set(ref))
        id = label + f"{prefix} " + str(i - 1)
        if i - 1 == 0:
            id = label + ": หัวเรื่อง"
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
        label + prefix + " " + x
        for x in re.findall(rf"{prefix} (\d+)", raw)
        if int(x) != i - 1
    ]
    ref_commasep = ",".join(set(ref))
    id = label + prefix + " " + str(i - 1)
    if i - 1 == 0:
        id = label + ": หัวเรื่อง"
    sections.append(
        Document(
            id=id,
            page_content=raw,
            metadata={"id": id, "ref": ref_commasep},
        )
    )
    return sections


def save_docs_to_jsonl(documents: t.Iterable[Document], file_path: str) -> None:
    with jsonlines.open(file_path, mode="w") as writer:
        for doc in documents:
            writer.write(doc.dict())


def split_csv(header, content_raw: str) -> list[Document]:
    docs = []
    data = csv.reader(content_raw.splitlines())
    head = []
    for i, row in enumerate(data):
        if i == 0:
            head = row
            continue
        content = (
            header["header"]
            + "\n"
            + "\n".join([f"{head[i]}: {x}" for i, x in enumerate(row)])
        )

        docs.append(
            Document(
                id=f"{header['name']} ({i})",
                page_content=content,
                metadata={"id": f"{header['name']} ({i})", "ref": ""},
            )
        )
    return docs


path = "./raw/"

# get files in the directory


files = os.listdir(path)

for file in files:
    if not file.endswith(".label.txt"):
        continue
    file_path = os.path.join(path, file)

    file_io = io.open(file_path, "r", encoding="utf-8")
    raws_nosplit = file_io.read()
    file_io.close()

    raws = raws_nosplit.split("---\n")
    all_splits = []
    for raw in raws:
        raw = raw.strip()
        json_start = raw.find("{")
        if json_start == -1:
            print(raw)
            raise ValueError("No JSON header found in the raw text")
        json_end = raw.find("}")
        if json_end == -1:
            print(raw)
            raise ValueError("No JSON header found in the raw text")

        label = raw[json_start : json_end + 1]
        content = raw[json_end + 1 :].strip()
        header = json.loads(label)

        assert "name" in header, "name is not in the header"
        if "mode" not in header:
            header["mode"] = "whole"

        if header["mode"] == "whole":
            doc = Document(
                id=header["name"],
                page_content=content,
                metadata={"id": header["name"], "ref": ""},
            )

            all_splits.append(doc)
            continue
        elif header["mode"] == "split":
            assert "prefix" in header, "prefix is not in the header"
            assert type(header["prefix"]) is str, "prefix is not a string"
            assert header["prefix"] != "", "prefix is empty"

            sections = law_section_splitter(header["name"], header["prefix"], content)
            all_splits.extend(sections)
            continue
        elif header["mode"] == "csv":
            assert "header" in header, "header is not in the header"
            assert type(header["header"]) is str, "header is not a string"
            assert header["header"] != "", "header is empty"

            rows = split_csv(header, content)
            all_splits.extend(rows)
        else:
            print(header)
            raise ValueError(f"Unknown mode {header['mode']}")

    save_docs_to_jsonl(all_splits, "./data/" + file.replace(".txt", ".jsonl"))
