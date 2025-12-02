from fastapi import FastAPI
import logging

from app.api.routes import router
from app.core.models import ModelManager

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
