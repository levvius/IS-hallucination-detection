# Implementation Guide - Remaining Tasks

## ✅ Completed (Phase 1 & 2 - Partial)

1. **Infrastructure**
   - ✅ Git branch created and pushed
   - ✅ requirements.txt updated (+12 dependencies)
   - ✅ Context7 documentation obtained and saved

2. **Exception System**
   - ✅ app/core/exceptions.py created (8 custom exceptions)
   - ✅ app/core/models.py updated with exceptions

3. **Security & Caching**
   - ✅ app/core/cache.py created (TTL cache, MD5 keys)
   - ✅ app/core/config.py updated (rate limiting settings)
   - ✅ app/api/schemas.py updated (XSS validation)

---

## 📋 TODO: Remaining Implementation

### Step 1: Update app/main.py

Add exception handlers and rate limiter initialization.

```python
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging

from app.api.routes import router
from app.core.models import ModelManager
from app.core.exceptions import (
    AppBaseException,
    ModelNotLoadedException,
    InputValidationException,
    KnowledgeBaseException
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Text Classification API",
    description="Classifies English text as truth, falsehood, or neutral using NLI and Wikipedia evidence",
    version="1.0.0"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# Exception handlers
@app.exception_handler(ModelNotLoadedException)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedException):
    """Handle model not loaded errors (503)."""
    logger.error(f"Model not loaded: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(KnowledgeBaseException)
async def knowledge_base_handler(request: Request, exc: KnowledgeBaseException):
    """Handle knowledge base errors (503)."""
    logger.error(f"Knowledge base error: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(InputValidationException)
async def input_validation_handler(request: Request, exc: InputValidationException):
    """Handle input validation errors (400)."""
    logger.warning(f"Input validation failed: {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.exception_handler(AppBaseException)
async def app_base_exception_handler(request: Request, exc: AppBaseException):
    """Handle all other application exceptions (500)."""
    logger.error(f"Application error: {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_dict()
    )


@app.on_event("startup")
async def startup_event():
    """Load models on application startup."""
    logger.info("Starting up application...")
    logger.info("Loading models...")

    try:
        mm = ModelManager.get_instance()
        mm.load_models()
        logger.info("✓ Models loaded successfully")
    except Exception as e:
        logger.error(f"✗ Failed to load models: {str(e)}", exc_info=True)
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on application shutdown."""
    logger.info("Shutting down application...")


# Include API routes
app.include_router(router, prefix="/api/v1", tags=["classification"])


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Text Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/v1/health"
    }
```

---

### Step 2: Update app/api/routes.py

Add rate limiting and caching to endpoints.

```python
from fastapi import APIRouter, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address
import logging

from app.api.schemas import ClassifyRequest, ClassifyResponse, HealthResponse
from app.services.classifier import classify_text
from app.core.models import ModelManager
from app.core.cache import get_cached_result, cache_result, get_cache_info
from app.core.exceptions import ClassificationException

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize limiter
limiter = Limiter(key_func=get_remote_address)


@router.post("/classify", response_model=ClassifyResponse)
@limiter.limit("10/minute")
async def classify_endpoint(request: Request, classify_req: ClassifyRequest):
    """
    Classify English text as truth, falsehood, or neutral.

    Rate limit: 10 requests per minute per IP.
    Cached results expire after 5 minutes.
    """
    try:
        # Check cache
        cached = get_cached_result(classify_req.text)
        if cached:
            logger.info("Cache hit - returning cached result")
            return cached

        # Classify text
        logger.info(f"Classifying text ({len(classify_req.text)} chars)")
        result = classify_text(classify_req.text)

        # Cache result
        cache_result(classify_req.text, result)

        return result

    except Exception as e:
        logger.error(f"Classification failed: {str(e)}", exc_info=True)
        raise ClassificationException(
            f"Failed to classify text: {str(e)}",
            details={"error_type": type(e).__name__}
        )


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.

    Returns service status and model loading state.
    """
    mm = ModelManager.get_instance()

    try:
        # Check if models are loaded
        _ = mm.get_embed_model()
        _ = mm.get_nli()
        _ = mm.get_index()
        snippets = mm.get_snippets()

        return HealthResponse(
            status="healthy",
            models_loaded=True,
            kb_size=len(snippets)
        )
    except Exception:
        return HealthResponse(
            status="not_ready",
            models_loaded=False,
            kb_size=0
        )


@router.get("/cache-info")
async def cache_info_endpoint():
    """Get cache statistics (for debugging)."""
    return get_cache_info()
```

---

### Step 3: Update Service Files with Exception Handling

**app/services/claim_extractor.py:**
```python
from app.core.exceptions import ClaimExtractionException

def extract_claims(text: str) -> List[str]:
    try:
        # ... existing code
        return claims
    except Exception as e:
        raise ClaimExtractionException(
            f"Failed to extract claims: {str(e)}",
            details={"text_length": len(text), "error": str(e)}
        )
```

**app/services/evidence_retriever.py:**
```python
from app.core.exceptions import EvidenceRetrievalException

def retrieve_proofs(claim: str, top_k: int = None) -> List[Dict]:
    try:
        # ... existing code
        return proofs
    except Exception as e:
        raise EvidenceRetrievalException(
            f"Failed to retrieve evidence: {str(e)}",
            details={"claim": claim[:50], "top_k": top_k, "error": str(e)}
        )
```

**app/services/nli_verifier.py:**
```python
from app.core.exceptions import NLIVerificationException

def nli_score(claim: str, evidence: str) -> float:
    try:
        # ... existing code
        return score
    except Exception as e:
        raise NLIVerificationException(
            f"NLI verification failed: {str(e)}",
            details={"claim": claim[:50], "evidence": evidence[:50], "error": str(e)}
        )
```

**app/services/classifier.py:**
```python
from app.core.exceptions import ClassificationException

def classify_text(text: str) -> Dict:
    try:
        # ... existing code
        return result
    except Exception as e:
        raise ClassificationException(
            f"Classification failed: {str(e)}",
            details={"text_length": len(text), "error": str(e)}
        )
```

---

### Step 4: Install Dependencies

```bash
source venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

---

### Step 5: Test the Changes

```bash
# Start the server
./run.sh

# In another terminal:
# Test health check
curl http://localhost:8000/api/v1/health

# Test classification
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Albert Einstein was born in 1879 in Germany."}'

# Test rate limiting (should get 429 after 10 requests)
for i in {1..15}; do
  curl -X POST http://localhost:8000/api/v1/classify \
    -H "Content-Type: application/json" \
    -d '{"text": "Test text for rate limiting."}' &
done

# Test XSS validation (should get 422)
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "<script>alert(1)</script>"}'
```

---

### Step 6: Create Testing Infrastructure (Optional - для полноты)

**pytest.ini:**
```ini
[pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts =
    -v
    --strict-markers
    --cov=app
    --cov-report=term-missing
    --cov-report=html
    --asyncio-mode=auto
markers =
    unit: Unit tests with mocks
    integration: Integration tests with real models
    slow: Slow tests (>1s)
    playwright: Playwright browser tests
asyncio_mode = auto
```

**.coveragerc:**
```ini
[run]
source = app
omit =
    */tests/*
    */venv/*
    */__init__.py

[report]
exclude_lines =
    pragma: no cover
    def __repr__
    raise AssertionError
    raise NotImplementedError
    if __name__ == .__main__.:
```

---

## Next Commit Message

После выполнения Steps 1-3:

```
feat: Add exception handlers, rate limiting, and routes integration

## Phase 2 Complete: API Integration

### Updated
- app/main.py: Exception handlers for all custom exceptions
  - ModelNotLoadedException (503)
  - KnowledgeBaseException (503)
  - InputValidationException (400)
  - AppBaseException (500)
  - Rate limiter initialization with slowapi

- app/api/routes.py: Rate limiting and caching
  - 10 requests/minute limit on /classify endpoint
  - Response caching with 5-minute TTL
  - Cache info endpoint for debugging
  - Improved error handling

- Services: Exception handling in all service layers
  - claim_extractor.py
  - evidence_retriever.py
  - nli_verifier.py
  - classifier.py

### Testing
- Validated exception handling
- Tested rate limiting (429 after limit)
- Verified XSS protection (422 for dangerous patterns)
- Confirmed caching behavior

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
```

---

## Summary

**Completed so far:**
- ✅ Exception system (8 exceptions)
- ✅ Caching system (TTL, MD5 keys)
- ✅ Config updates (rate limiting)
- ✅ ModelManager with exceptions
- ✅ Schemas with XSS validation

**Remaining (Priority):**
1. Update main.py (exception handlers + rate limiter)
2. Update routes.py (rate limiting + caching)
3. Update services (exception wrapping)
4. Test manually
5. Commit Phase 2

**Optional (Later):**
- Create pytest infrastructure
- Write unit tests
- Write integration tests
- Write Playwright tests
- Create CI/CD pipeline
- Update documentation
