import asyncio
import csv
import json
import logging
from os import getenv

import aiofiles
from dotenv import load_dotenv
from langchain.load import dumps, loads
from langchain.prompts import PromptTemplate
from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import (
    CrossEncoderReranker,
)
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.documents.base import Document
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from qdrant_client import QdrantClient

from embedder import Embeddings
from reasoning_chat import ChatReasoning
from tools import calculate_tax
from vectorstore import doc_kv, retriever

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


def env(key: str) -> str:
    data = getenv(key)
    assert data is not None
    return data


llm = ChatReasoning(
    # model="gpt-4o-mini",
    # base_url="http://127.0.0.1:6000/v1",
    api_key=SecretStr(env("OPENROUTER_API_KEY")),
    base_url=env("OPENROUTER_BASE_URL"),
    model="qwen/qwq-32b",
    temperature=0,
    timeout=None,
    max_retries=10,
    extra_body={"include_reasoning": True},
)

# llm_cot = ChatOpenAI(
#     model="gpt-4o-mini",
#     # base_url="http://127.0.0.1:6000/v1",
#     # extra_body={"optillm_approach": "cot_reflection"},
# )
llm_cot = llm

fast_llm = ChatOpenAI(
    # model="gpt-4o-mini",
    # base_url="http://127.0.0.1:6000/v1",
    api_key=SecretStr(env("OPENROUTER_API_KEY")),
    base_url=env("OPENROUTER_BASE_URL"),
    model="qwen/qwen-turbo",
    temperature=0,
    timeout=None,
    max_retries=10,
)

tools = [calculate_tax]
llm_with_tools = llm.bind_tools(tools)

embedder = Embeddings()

# create a Chroma object from the persisted directory
# vectorstore = Chroma(
#     persist_directory="./chroma_db",
#     embedding_function=embedder,  # type: ignore
# )


# sparse_embeddings = FastEmbedSparse(model_name="Qdrant/bm25")
# client = QdrantClient(path="./qdrant")


# vectorstore = QdrantVectorStore(
#     client=client,
#     collection_name="data",
#     embedding=embedder,
#     sparse_embedding=sparse_embeddings,
#     sparse_vector_name="sparse",
#     retrieval_mode=RetrievalMode.HYBRID,
# )
# retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 20})

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
            """You are an legal assistant that answer questions about the Public Procurement and Asset Management Act for government. A very important role in the government's ministry where you must answer with high accuracy that can be relied on.
            
Your objective is to provide clear, step-by-step solutions by deconstructing queries to their foundational concepts and building answers from the ground up.

Use the following pieces of retrieved context as a evidence and reference to your reasoning and answering. You must convert any Thai numerals or thai word that mean the number to Arabic numerals. You can ask for more explanation to get better understanding. And please answer with politeness manner for Thai people with male-based pronouns (ครับ).

    + Only derive the knowledge and the answer from the provided context and do not provide any information that is not in the context. SO DO NOT HALLUCINATE A RANDOM INFORMATION THAT IS NOT IN THE CONTEXT.
    + Before you answer the question, please read the context below carefully. If you need more information, you can ask for more explanation.
    + Context contents that are inside "[]" means that it is a data that later added to be more informative to you the AI, which is not a part of the original content, but **You should use it to answer the question**.
    + This year is พ.ศ. {year_th}, data or question that related to time might get changed due to postponded or delayed events. Please use the current year as a reference only if needed.
    + You must always provided all sources of the answer (label as "Label/Origin") at the end, give it in specific name that you can refer to the source later.
    + Since all data is a legal document in some way, you must think about how it interact with each other, does one line will disable the usage of the other, can 2 lines be done together.
    + การจัดซื้อจัดจ้างสิ่งต่างๆ นั้น สามารถทำได้ 3 วิธี ซึ่งเป็นคนละอย่างกัน ได้แก่ 1. วิธีประกาศเชิญชวนทั่วไป 2. วิธีคัดเลือก 3. วิธีเฉพาะเจาะจง
    + When you found a thai word that could convert to number, you must reciting this table EVERYTIME, do not skip it. And you must show the original thai word and the converted number in the answer like "2,000 บาท (สองพันบาท)".
    Thai numeric table:
    10 = สิบ
    100 = ร้อย
    1,000 = พัน
    10,000 = หมื่น
    100,000 = แสน
    1,000,000 = ล้าน
    10,000,000 = ล้าน
    ...
    

""",
        ),
        ("system", "Context: {context}"),
        ("human", "{question}"),
    ]
)


from datetime import datetime

year_th = datetime.now().year + 543


rewrite_prompt = PromptTemplate(
    input_variables=["question"],
    template=f"""You are an AI language model assistant. Your task is
    to generate sentences of important keywords or phrases that is relevant to the user question to retrieve relevant Thai legal documents like act of legislation from a vector database.
    By generating multiple perspectives on the user question,
    your goal is to help the user overcome some of the limitations
    of distance-based similarity search. Provide these alternative
    sentence, separated by newlines without any kind of suffix (number, dash, etc) and only the your answer without any header **in Thai**.

    You should generate using these strategies, one for each if not specified:
        1. Step back and paraphrase a question to a more generic step-back question, which is easier to answer.
        2. Provide a question that is more specific or detailed than the original question.
        3. Paraphrase the question to be easier to find.
        4. Paraphrase the question in opposite manner, e.g. if the question is asking "Is it an A", try paraphrase is to "Is it not A".
        5. If the sentence is composed of multiple parts, provide a question that focuses on one of those parts. If the sentence has only one part, ignore this strategy.

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


def split_query(text: str) -> list[str]:
    raw = [v.strip() for v in text.split("\n")]
    res = []
    for v in raw:
        if v:
            print("\t- split_query", v)
            res.append(v)

    return res


generate_query = rewrite_prompt | fast_llm | StrOutputParser() | split_query


def reciprocal_rank_fusion(results: list[list[Document]], k=60) -> list[Document]:
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


def recursive_query(ref: list[str], done: set[str], result: list[Document]):
    if not ref:
        return

    related_docs = doc_kv.mget(ref)

    if not related_docs:
        return
    new_ref = []
    for i, doc in enumerate(related_docs):
        if doc is None:
            print("DOC NOT FOUND!!!", ref[i])
            continue
        result.append(doc)
        ref_doc = doc.metadata.get("ref")
        if ref_doc:
            new_ref.extend(ref_doc.split(","))
    done.update(ref)
    new_ref = list(set(new_ref).difference(done))
    recursive_query(new_ref, done, result)


def get_related_docs(docs: list[Document]):
    result = []
    refs = []
    for doc in docs:
        result.append(doc)
        ref = doc.metadata.get("ref")
        if ref:
            refs.extend(ref.split(","))

    recursive_query(refs, set(), result)
    # dedupe result by section metadata
    return result


def format_docs(docs: list[Document]):
    datas = []
    used = set()
    for doc in docs:
        if doc.metadata.get("id") in used:
            continue
        used.add(doc.metadata.get("id"))
        print(doc.metadata.get("id"))
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
    | llm_cot
    # | StrOutputParser()
)


# Function to process each row with semaphore
async def process_row(sem, row, res, no):
    async with sem:
        question = row[0]

        # if level != "พื้นฐาน":
        #     return  # Skip if level is not "พื้นฐาน"

        expected_answer = row[1]
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
            # answer = BaseMessage("Error while processing the question")
            return

        # Append result to the shared list
        res.append(
            {
                "number": no,
                "question": question,
                "expected": expected_answer,
                "answerFromLLM": answer.content,
            }
        )


from datetime import datetime


# Main function to read CSV and process rows concurrently
async def main():
    res = []
    sem = asyncio.Semaphore(
        10
    )  # Adjust this number based on how many tasks you want to run concurrently

    async with aiofiles.open("question-set.csv", mode="r", encoding="utf-8") as f:
        csv_reader = csv.reader(await f.readlines())
        next(csv_reader)  # Skip the header

        tasks = []
        for i, row in enumerate(csv_reader, 1):
            tasks.append(process_row(sem, row, res, i))
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

# Run the async main loop
if __name__ == "__main__":
    MODE = sys.argv[1] if len(sys.argv) > 1 else "terminal"
    match MODE:
        case "terminal":
            while True:
                inp = input("You: ")
                if inp == "exit":
                    break
                messages: list[BaseMessage] = [HumanMessage(inp)]

                print(messages, "\n")
                print("Chatbot:")
                reason_end = 1
                for msg in rag_chain.stream(messages):
                    content = msg.content
                    reason = msg.additional_kwargs.get("reasoning") or ""
                    if not reason:
                        if reason_end == 1:
                            reason_end = 0
                        else:
                            reason_end = -1
                    if reason_end == 0:
                        print()
                        print("REASON END")

                    print(reason, end="", flush=True)
                    print(content, end="", flush=True)

                print()
                # print(
                #     "Chatbot:",
                #     rag_chain.invoke(messages).content,
                #     "\n",
                # )

        case "file":
            asyncio.run(main())
