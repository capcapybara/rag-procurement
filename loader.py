from langchain_community.document_loaders import UnstructuredPDFLoader

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
    docs.append(doc)


with open("data.json", "w") as f:
    for doc in docs:
        f.write(doc.json() + "\n")
