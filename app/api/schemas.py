from pydantic import BaseModel, Field, field_validator
from typing import List, Optional
import re


class ClassifyRequest(BaseModel):
    """Request schema for text classification."""
    text: str = Field(
        ...,
        min_length=10,
        max_length=5000,
        description="English text to classify"
    )

    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """
        Validate input text for security and quality.

        Checks:
        - Minimum word count (3 words)
        - XSS/injection patterns
        - Whitespace normalization

        Args:
            v: Input text to validate

        Returns:
            Normalized text

        Raises:
            ValueError: If validation fails
        """
        # Check minimum word count
        words = v.split()
        if len(words) < 3:
            raise ValueError('Text must contain at least 3 words')

        # Check for dangerous XSS/injection patterns
        dangerous_patterns = [
            (r'<script[^>]*>.*?</script>', 'script tag'),
            (r'javascript:', 'javascript protocol'),
            (r'onerror\s*=', 'onerror attribute'),
            (r'onclick\s*=', 'onclick attribute'),
            (r'onload\s*=', 'onload attribute'),
            (r'<iframe[^>]*>', 'iframe tag'),
            (r'eval\s*\(', 'eval function'),
            (r'document\.cookie', 'cookie access'),
            (r'<embed[^>]*>', 'embed tag'),
            (r'<object[^>]*>', 'object tag'),
        ]

        for pattern, name in dangerous_patterns:
            if re.search(pattern, v, re.IGNORECASE | re.DOTALL):
                raise ValueError(f'Suspicious pattern detected: {name}')

        # Normalize whitespace
        v = ' '.join(v.split())

        return v


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
