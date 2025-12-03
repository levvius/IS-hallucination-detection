import faiss
from typing import List, Dict

from app.core.models import ModelManager
from app.core.config import settings
from app.core.exceptions import EvidenceRetrievalException


def retrieve_proofs(claim: str, top_k: int = None) -> List[Dict[str, any]]:
    """
    Retrieve evidence snippets for a claim using FAISS similarity search.

    Args:
        claim: The claim text to find evidence for
        top_k: Number of top evidence snippets to return. If None, uses settings.top_k_proofs

    Returns:
        List of dicts with keys: snippet, source, retrieval_score

    Raises:
        EvidenceRetrievalException: If evidence retrieval fails
    """
    try:
        if top_k is None:
            top_k = settings.top_k_proofs

        # Get model manager
        mm = ModelManager.get_instance()
        embed_model = mm.get_embed_model()
        index = mm.get_index()
        kb_docs = mm.get_snippets()

        # Encode claim
        emb = embed_model.encode([claim], convert_to_numpy=True)

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(emb)

        # Search FAISS index
        D, I = index.search(emb, top_k)

        # Build results
        results = []
        for i, score in zip(I[0], D[0]):
            results.append({
                "snippet": kb_docs[i]["snippet"],
                "source": kb_docs[i]["source"],
                "retrieval_score": float(score)
            })

        return results

    except Exception as e:
        raise EvidenceRetrievalException(
            f"Failed to retrieve evidence: {str(e)}",
            details={"claim": claim[:50], "top_k": top_k, "error": str(e)}
        )
