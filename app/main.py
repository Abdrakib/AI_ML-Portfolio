"""FastAPI application for brain tumor classification."""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from  config import MODEL_PATH, LABEL_MAP_PATH
from app.core.model_loader import init_model
from app.api.routes import predict, health

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup."""
    try:
        init_model(MODEL_PATH, LABEL_MAP_PATH)
        logger.info("Model and label map loaded successfully")
    except FileNotFoundError as e:
        logger.warning("Could not load model at startup: %s", e)
        logger.warning("Place model at %s and label map at %s", MODEL_PATH, LABEL_MAP_PATH)
    yield
    # Shutdown: nothing to do


app = FastAPI(
    title="Brain Tumor Classification API",
    description="Classify brain MRI images and return Grad-CAM overlay.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(predict.overlay_router)
