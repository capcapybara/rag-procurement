import asyncio
import csv
import json
import logging

import aiofiles
from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain.prompts import PromptTemplate
from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_chroma import Chroma
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.documents.base import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

from embedder import Embeddings

logging.basicConfig()
logging.getLogger("langchain.retrievers.multi_query").setLevel(logging.INFO)

load_dotenv()

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

n_gpu_layers = 1
n_batch = 512

# Make sure the model path is correct for your system!
# llm: Type[BaseLLM] = LlamaCpp(
#     model_path="./SeaLLM3-7B-Chat.Q5_K_M.gguf",
#     temperature=0.75,
#     max_tokens=2000,
#     top_p=1,
#     n_gpu_layers=n_gpu_layers,
#     n_batch=n_batch,
#     f16_kv=True,
#     n_ctx=50000,
#     callback_manager=callback_manager,
#     # verbose=True,
# )

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

embedder = Embeddings()

# create a Chroma object from the persisted directory
vectorstore = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedder,  # type: ignore
)

retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

step_back_example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant for question-answering tasks that designed to answer about land and real estate law in Bangkok. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know,. You must convert any Thai numerals or thai word that mean the number to Arabic numerals. You can ask for more explanation to get better understanding. And please answer with politeness manner for Thai people with male-based pronouns (ครับ).

    + Before you answer the question, please read the context below carefully. If you need more information, you can ask for more explanation.
    + Context contents that are inside "[]" means that it is a data that later added to be more informative to you the AI, which is not a part of the original content. **You must use it to answer the question**.
    + This year is พ.ศ. {year_th}, data or question that related to time might get changed due to postponded or delayed events. Please use the current year as a reference only if needed.
    + You must always provided all sources of the answer (label as "Label/Origin") at the end, give it in specific name that you can refer to the source later.

    Foundational knowledge:
    + การชำระภาษีที่ดินและสิ่งปลูกสร้าง ต้องคำนึงถึงทั้งตัวที่ดินเองและสิ่งปลูกสร้างที่ตั้งอยู่บนที่ดินนั้น
    เช่น ถ้าหากมีที่ดินและสิ่งปลูกสร้างมีเจ้าของเป็นคนเดียว จะต้องนำราคาของทั้งสองอย่างมารวมกันแล้วคำนวณภาษี แต่ถ้าหากมีเจ้าของที่ดินและสิ่งปลูกสร้างเป็นคนละคนกัน จะต้องคำนึงถึงราคาของที่ดินและสิ่งปลูกสร้างแยกกันตามเจ้าของ ซึ่งก็เป็นไปได้ว่าเจ้าของที่ดินจะต้องเสียภาษีแม้ว่าจะสิ่งปลูกสร้างดังกล่าวจะไม่ต้องเสียภาษี
    + กรณีที่มีการยกเว้นภาษีที่ดินและสิ่งปลูกสร้าง จะไม่ต้องคำนึงถึงปัจจัยอื่นๆ นอกจากปัจจัยที่ระบุไว้ในกฎหมายที่พูดถึงการยกเว้นภาษีนั้น
""",
        ),
        ("system", "Context: {context}"),
        ("human", "{question}"),
    ]
)

# prompt.add_message("system", system_prompt)

from datetime import datetime

year_th = datetime.now().year + 543


rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template=f"""You are an AI language model assistant. Your task is
    to generate sentences of important keywords or phrases that is relevant to the user question to retrieve relevant Thai legal documents like act of legislation from a vector database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    sentence. separated by newlines without any kind of suffix (number, dash, etc).

    You should generate using these strategies, one for each if not specified:
        1. Step back and paraphrase a question to a more generic step-back question, which is easier to answer.
        2. Provide a question that is more specific or detailed than the original question.
        3. Paraphrase the question to be easier to find.
        4. Paraphrase the question in opposite manner, e.g. if the question is asking "Is it an A", try paraphrase is to "Is it not A".
        5. If the sentence is composed of multiple parts, provide a question that focuses on one of those parts. If the sentence has only one part, ignore this strategy.
    
    Known opposite words:
    - เสียภาษี : ยกเว้นภาษี


    And this year is พ.ศ. {year_th}, data or question that related to time might get changed due to postponded or delayed events. Please use the current year as a reference, do not use any relative time question at all.

    Original question: {"{"}question{"}"}""",
)
# rewrite_prompt = ChatPromptTemplate.from_template(template)


def rewrite_parse(text: str):
    print("rewrite_parse", text)
    return text.strip('"').strip("**")


# rewriter = rewrite_prompt | llm | StrOutputParser() | rewrite_parse


# multi_query_retriever = MultiQueryRetriever.from_llm(
#     retriever=retriever,
#     llm=llm,
#     prompt=rewrite_prompt,
#     include_original=True,
# )


def split_query(text: str):
    raw = [v.strip() for v in text.split("\n")]
    res = []
    for v in raw:
        if v:
            print("\t- split_query", v)
            res.append(v)

    return res


generate_query = rewrite_prompt | llm | StrOutputParser() | split_query


def reciprocal_rank_fusion(results: list[list], k=60):
    fused_scores = {}
    for docs in results:
        # Assumes the docs are returned in sorted order of relevance
        for rank, doc in enumerate(docs):
            doc_str = dumps(doc)
            if doc_str not in fused_scores:
                fused_scores[doc_str] = 0
            previous_score = fused_scores[doc_str]
            fused_scores[doc_str] += 1 / (rank + k)

    reranked_results = [
        loads(doc)
        for doc, _ in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    ]
    return reranked_results[:20]


multi_query_retriever = generate_query | retriever.map() | reciprocal_rank_fusion


crossencoder = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-v2-m3")
compressor = CrossEncoderReranker(model=crossencoder, top_n=20)
# compression_retriever = ContextualCompressionRetriever(
#     base_compressor=compressor, base_retriever=multi_query_retriever
# )


def recursive_query(ref: list[str], result: list[Document]):
    if not ref:
        return

    ref_filter = {"id": {"$in": ref}}
    related_docs = vectorstore.similarity_search(
        query="",
        filter=ref_filter,  # type: ignore
    )

    if not related_docs:
        return
    new_ref = []
    for doc in related_docs:
        result.append(doc)
        ref_doc = doc.metadata.get("ref")
        if ref_doc:
            new_ref.extend(ref_doc.split(","))
    recursive_query(new_ref, result)


def get_related_docs(docs):
    result = []
    refs = []
    for doc in docs:
        result.append(doc)
        ref = doc.metadata.get("ref")
        if ref:
            refs.extend(ref.split(","))

    recursive_query(refs, result)
    # dedupe result by section metadata
    return result


def format_docs(docs: list[Document]):
    datas = []
    for doc in docs:
        data = f"""
Label/Origin: {doc.metadata.get("id")}
Content: {doc.page_content}
"""
        datas.append(data)
    res = "\n---\n\n".join(datas)

    print(f"Estimated total length: {len(res)/5}")

    return res


rag_chain = (
    {
        "year_th": lambda _: year_th,
        "context": multi_query_retriever | get_related_docs | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# Function to process each row with semaphore
async def process_row(sem, row, res):
    async with sem:
        no = row[0]
        question = row[1]
        level = row[2]

        if level != "พื้นฐาน":
            return  # Skip if level is not "พื้นฐาน"

        expected_answer = row[3]
        print(f"Question {no}: {question}")

        answer = ""
        for i in range(5):
            try:
                answer = await rag_chain.ainvoke({"question": question})
                break
            except Exception as e:
                print(f"Retrying question {no}: {question}")
                print(e)
                await asyncio.sleep(3**i)

        if not answer:
            print(f"Error while processing question {no}: {question}")
            answer = "Error while processing the question"

        # Append result to the shared list
        res.append(
            {
                "number": no,
                "question": question,
                "level": level,
                "expected": expected_answer,
                "answerFromLLM": answer,
            }
        )


from datetime import datetime


# Main function to read CSV and process rows concurrently
async def main():
    res = []
    sem = asyncio.Semaphore(
        3
    )  # Adjust this number based on how many tasks you want to run concurrently

    async with aiofiles.open(
        "question_simple_section.csv", mode="r", encoding="utf-8"
    ) as f:
        csv_reader = csv.reader(await f.readlines())
        next(csv_reader)  # Skip the header

        tasks = []
        for row in csv_reader:
            tasks.append(process_row(sem, row, res))

        await asyncio.gather(*tasks)
        res.sort(key=lambda x: int(x["number"]))
        with open(
            f"./result/result-{datetime.now().strftime('%Y-%m-%d-%H:%M:%S')}.json",
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            json.dump(res, f, indent=4, ensure_ascii=False)


# get the mode argument
import sys

MODE = sys.argv[1] if len(sys.argv) > 1 else "terminal"

# Run the async main loop
if __name__ == "__main__":
    match MODE:
        case "terminal":
            while True:
                inp = input("You: ")
                if inp == "exit":
                    break
                print("Chatbot:", rag_chain.invoke(inp))
        case "file":
            asyncio.run(main())
        case _:
            print("Invalid mode")
