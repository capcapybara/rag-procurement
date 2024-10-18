import asyncio
import json

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI

load_dotenv()

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
)

prompt = PromptTemplate.from_template(
    """I have an extracted data from pdf file with a lot of errors and typo.
    I want to clean it up. Please only return the cleaned data.

Data: {question}

Cleaned:"""
)

rag_chain = (
    {
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)
res = []


async def process_data(data, i):
    print(f"Work {i + 1}")
    s = await rag_chain.ainvoke(data)

    res[i] = s
    print(f"Done {i + 1}")


with open("./raw/dirty.json", "r") as dirty:
    data = json.load(dirty)
    res = [None] * len(data)

    async def main():
        wrks = []
        for i, d in enumerate(data):
            wrks.append(process_data(d, i))

        await asyncio.gather(*wrks)

    asyncio.run(main())


with open("./raw/cleaned.json", "w") as cleaned:
    json.dump(res, cleaned, indent=4, ensure_ascii=False)
