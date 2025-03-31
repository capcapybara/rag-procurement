from collections import defaultdict
from typing import List

import langchain_core.embeddings as lang
import torch
from langchain_qdrant import SparseEmbeddings as QDrantSparseEmbeddings
from langchain_qdrant import SparseVector
from sentence_transformers import SentenceTransformer
from transformers import AutoModel, AutoTokenizer

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
