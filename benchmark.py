import asyncio
import json
import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from os import getenv

load_dotenv()

# llm = ChatOpenAI(
#     model="gpt-4o-mini",
#     temperature=0,
#     timeout=None,
#     max_retries=10,
# )


def env(key: str) -> str:
    data = getenv(key)
    assert data is not None
    return data


llm = ChatOpenAI(
    api_key=SecretStr(env("OPENROUTER_API_KEY")),
    base_url=env("OPENROUTER_BASE_URL"),
    model="qwen/qwen-turbo",
    temperature=0,
    timeout=None,
    max_retries=10,
)

prompt = PromptTemplate.from_template(
    """I have a system that answers questions. I want to benchmark it by
checking the correctness of the answers compared to the expected answers.
Please only return the result of the benchmark in -1 and 1, 1 means the answer is correct with the same meaning as expected, and -1 means the answer is incorrect or not relevant to the question.

Question: {question}

Expected: {correct}

Result: {result}
"""
)

rag_chain = (
    {
        "question": RunnablePassthrough(),
        "correct": RunnablePassthrough(),
        "result": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
res = []


async def process_data(data, i, sem):
    async with sem:
        s = ""
        for retry in range(3):
            try:
                s = await rag_chain.ainvoke(
                    input={
                        "question": data["question"],
                        "correct": data["expected"],
                        "result": data["answerFromLLM"],
                    }
                )
                break
            except Exception as e:
                print(f"Retrying work {i} {retry}")
                print(e)
                await asyncio.sleep(3**retry)

        if not s:
            s = "Error"

        data["result"] = s

        res[i] = data
        print(f"Done {i + 1}")


files = os.listdir("./result")
exists = os.listdir("./result_bench")


sem = asyncio.Semaphore(
    3
)  # Adjust this number based on how many tasks you want to run concurrently

for file in files:
    if file in exists:
        continue
    print(f"Processing {file}")
    with open(f"./result/{file}", "r") as dirty:
        data = json.load(dirty)
        res = [None] * len(data)

        async def main():
            wrks = []
            for i, d in enumerate(data):
                wrks.append(process_data(d, i, sem))

            await asyncio.gather(*wrks)

        asyncio.run(main())

    with open(f"./result_bench/{file}", "w") as cleaned:
        json.dump(res, cleaned, indent=4, ensure_ascii=False)
