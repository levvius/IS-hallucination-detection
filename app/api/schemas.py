from pydantic import BaseModel, Field
from typing import List, Optional


class ClassifyRequest(BaseModel):
    """Request schema for text classification."""
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="English text to classify"
    )


class ProofEvidence(BaseModel):
    """Evidence snippet with scores."""
    snippet: str
    source: str
    nli_score: float
    retrieval_score: float


class ClaimAnalysis(BaseModel):
    """Analysis result for a single claim."""
    claim: str
    classification: str = Field(
        ...,
        description='Classification result: "правда", "неправда", or "нейтрально"'
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    best_evidence: Optional[ProofEvidence] = None


class ClassifyResponse(BaseModel):
    """Response schema for text classification."""
    overall_classification: str = Field(
        ...,
        description='Overall classification: "правда", "неправда", or "нейтрально"'
    )
    confidence: float = Field(..., ge=0.0, le=1.0)
    claims: List[ClaimAnalysis]


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: bool
    kb_size: int
