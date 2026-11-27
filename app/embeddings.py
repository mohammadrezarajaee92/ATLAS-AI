from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import List, Protocol

from app.config import settings

logger = logging.getLogger(__name__)

settings = settings()

class EmbeddingProvider(Protocol):
    model_name: str
    dimension: int

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        ...

    def embed_query(self, text: str) -> List[float]:
        ...

@dataclass
class EmbeddingInfo:
    provider: str
    model_name: str
    dimension: int
    distance: str = "cosine"

    @property
    def collection_suffix(self) -> str:
        safe_model = (
            self.model_name.replace("/", "_")
            .replace("-", "_")
            .replace(".", "_")
            .lower()
        )
        return f"{self.provider}_{safe_model}_{self.dimension}"



class GeminiEmbeddingProvider:
    def __init__(self):
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        self.model_name = settings.embedding_model
       
        self._client = GoogleGenerativeAIEmbeddings(
            model=self.model_name,
            google_api_key=settings.google_api_key,
        )
        
        
        self.dimension = getattr(settings, "embedding_dimention", 3072)
        logger.info(f"Gemini Provider Ready: {self.model_name} | Dim: {self.dimension}")

    def _retry(self, fn, *args, **kwargs):
        max_retries = 5
        for attempt in range(max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                msg = str(e).lower()
               
                retryable = ["429", "timeout", "deadline", "503", "504", "exhausted"]
                if any(x in msg for x in retryable) and attempt < max_retries:
                    wait = (2 ** attempt) + random.random()
                    logger.warning(f"Retry {attempt+1}/{max_retries} in {wait:.1f}s due to: {e}")
                    time.sleep(wait)
                    continue
                raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        if not texts: return []
        
    
        internal_batch_size = 2 
        all_vecs = []
        
        for i in range(0, len(texts), internal_batch_size):
            batch = texts[i : i + internal_batch_size]
            vecs = self._retry(self._client.embed_documents, batch)
            all_vecs.extend(vecs)
            
            
            if i + internal_batch_size < len(texts):
                time.sleep(2) 
        
        return all_vecs

    def embed_query(self, text: str) -> List[float]:
        return self._retry(self._client.embed_query, text)



def get_embedding_provider() -> EmbeddingProvider:
    provider = settings.embedding_provider.lower().strip()

    if provider == "gemini":
        return GeminiEmbeddingProvider()
    else:
        raise ValueError(
            f"Unsupported EMBEDDING_PROVIDER={settings.embedding_provider!r}. "
            "Use one of: 'gemini', 'local'"
        )


def get_embedding_info(provider: EmbeddingProvider) -> EmbeddingInfo:
    provider_name = settings.embedding_provider.lower().strip()
    return EmbeddingInfo(
        provider=provider_name,
        model_name=provider.model_name,
        dimension=provider.dimension,
        distance="cosine",
    )
