from typing import List

from sentence_transformers import SentenceTransformer
import langchain_core.embeddings as lang

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class Embeddings(lang.Embeddings):
    def __init__(self, model=DEFAULT_MODEL):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    async def aembed_query(self, text: str) -> List[float]:
        return self.model.encode([text]).tolist()[0]
