import logging
from datetime import datetime
from typing import Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from scalar_fastapi import get_scalar_api_reference
from scalar_fastapi.scalar_fastapi import Theme

from app.api.chat import router as chat_router
from app.config import settings
from app.models import HealthResponse

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper()),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Life Insurance Support Assistant API",
    description="AI-powered assistant for life insurance queries using LangGraph and RAG",
    version="1.0.0",
    docs_url=None,
    redoc_url=None,
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router)


@app.get("/", tags=["root"])
async def root() -> Dict[str, str]:
    return {
        "message": "Life Insurance Support Assistant API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/docs", include_in_schema=False)
async def scalar_html() -> HTMLResponse:
    return get_scalar_api_reference(
        openapi_url=app.openapi_url,
        title=app.title,
        theme=Theme.BLUE_PLANET,
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy", environment=settings.environment, timestamp=datetime.utcnow()
    )


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Starting Life Insurance Support Assistant API")
    logger.info(f"Environment: {settings.environment}")
    logger.info(f"LLM Model: {settings.llm_model}")


@app.on_event("shutdown")
async def shutdown_event() -> None:
    logger.info("Shutting down Life Insurance Support Assistant API")


# Main entry point
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.api_reload,
    )
