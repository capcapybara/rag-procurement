from langchain_chroma import Chroma

from embedder import Embeddings

embedder = Embeddings("Alibaba-NLP/gte-multilingual-base")

# create a Chroma object from the persisted directory
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedder,
)


retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 6})

retrieved_docs = retriever.invoke("คณะอนุกรรมการประจำจังหวัด")

len(retrieved_docs)
for doc in retrieved_docs:
    print("---")
    print(doc)
