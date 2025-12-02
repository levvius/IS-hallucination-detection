from fastapi import APIRouter, HTTPException
import logging

from app.api.schemas import (
    ClassifyRequest,
    ClassifyResponse,
    HealthResponse
)
from app.services.classifier import classify_text
from app.core.models import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/classify", response_model=ClassifyResponse)
async def classify_endpoint(request: ClassifyRequest):
    """
    Classify text as "правда" (truth), "неправда" (falsehood), or "нейтрально" (neutral).

    Args:
        request: ClassifyRequest with text field

    Returns:
        ClassifyResponse with overall classification, confidence, and claim analysis
    """
    try:
        logger.info(f"Classifying text (length: {len(request.text)})")
        result = classify_text(request.text)
        logger.info(f"Classification result: {result['overall_classification']}")
        return result
    except Exception as e:
        logger.error(f"Classification error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns:
        HealthResponse with status, models_loaded flag, and KB size
    """
    try:
        mm = ModelManager.get_instance()
        models_loaded = mm._embed_model is not None
        kb_size = len(mm._kb_snippets) if mm._kb_snippets else 0

        return {
            "status": "healthy" if models_loaded else "not_ready",
            "models_loaded": models_loaded,
            "kb_size": kb_size
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
