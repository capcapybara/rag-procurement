from typing import List

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class Embeddings:
    def __init__(self, model=DEFAULT_MODEL):
        self.model = SentenceTransformer(model, trust_remote_code=True)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]

    def embed_query(self, query: str) -> List[float]:
        return self.model.encode([query]).tolist()[0]
