import json
import logging
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import faiss

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Singleton class for managing ML models and FAISS index.
    Ensures models are loaded once and reused across requests.
    """
    _instance: Optional['ModelManager'] = None
    _embed_model: Optional[SentenceTransformer] = None
    _nli_pipeline = None
    _faiss_index: Optional[faiss.Index] = None
    _kb_snippets: Optional[List[Dict[str, str]]] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_instance(cls) -> 'ModelManager':
        """Get the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def load_models(self):
        """Load all models and FAISS index."""
        if self._embed_model is not None:
            logger.info("Models already loaded")
            return

        logger.info("Loading models...")

        # Load embedding model
        logger.info(f"Loading embedding model: {settings.embedding_model}")
        self._embed_model = SentenceTransformer(settings.embedding_model)

        # Load NLI pipeline
        logger.info(f"Loading NLI model: {settings.nli_model}")
        device = -1 if settings.device == "cpu" else 0
        self._nli_pipeline = pipeline(
            "text-classification",
            model=settings.nli_model,
            device=device
        )

        # Load FAISS index
        logger.info(f"Loading FAISS index from: {settings.faiss_index_path}")
        if not settings.faiss_index_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found at {settings.faiss_index_path}. "
                f"Please run 'python scripts/build_kb.py' first."
            )
        self._faiss_index = faiss.read_index(str(settings.faiss_index_path))

        # Load KB snippets metadata
        logger.info(f"Loading KB snippets from: {settings.kb_snippets_path}")
        if not settings.kb_snippets_path.exists():
            raise FileNotFoundError(
                f"KB snippets not found at {settings.kb_snippets_path}. "
                f"Please run 'python scripts/build_kb.py' first."
            )
        with open(settings.kb_snippets_path, "r", encoding="utf-8") as f:
            self._kb_snippets = json.load(f)

        logger.info(f"Models loaded successfully. KB size: {len(self._kb_snippets)} snippets")

    def get_embed_model(self) -> SentenceTransformer:
        """Get the embedding model."""
        if self._embed_model is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._embed_model

    def get_nli(self):
        """Get the NLI pipeline."""
        if self._nli_pipeline is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._nli_pipeline

    def get_index(self) -> faiss.Index:
        """Get the FAISS index."""
        if self._faiss_index is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._faiss_index

    def get_snippets(self) -> List[Dict[str, str]]:
        """Get the KB snippets metadata."""
        if self._kb_snippets is None:
            raise RuntimeError("Models not loaded. Call load_models() first.")
        return self._kb_snippets
