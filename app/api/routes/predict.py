"""Prediction routes."""
import logging
from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import Response

from core.predictor import predict_with_gradcam, get_gradcam_overlay_bytes

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("")
async def predict(file: UploadFile = File(...)):
    """
    Classify brain MRI image and return prediction with Grad-CAM overlay.
    Returns: pred_label, confidence, probs, gradcam_overlay_b64 (base64 PNG).
    """
    logger.info("POST /predict received request")
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning("Invalid file type: %s", file.content_type)
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Expected image (JPEG, PNG, etc.).",
        )

    try:
        img_bytes = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to read image file.") from e

    if len(img_bytes) == 0:
        logger.warning("Empty image file received")
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        result = predict_with_gradcam(img_bytes)
        logger.info("Prediction success: %s", result.get("pred_label"))
        return result
    except RuntimeError as e:
        if "not loaded" in str(e):
            logger.error("Model not loaded: %s", e)
            raise HTTPException(
                status_code=503,
                detail="Model not ready. Ensure model and label map are loaded at startup.",
            ) from e
        logger.exception("RuntimeError in prediction: %s", e)
        raise
    except ValueError as e:
        logger.warning("ValueError in prediction: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed: %s", e)
        raise HTTPException(status_code=500, detail="Prediction failed.") from e


overlay_router = APIRouter(tags=["predict"])


@overlay_router.post("/predict_overlay")
async def predict_overlay(file: UploadFile = File(...)):
    """
    Return the Grad-CAM overlay as PNG directly (no prediction metadata).
    """
    logger.info("POST /predict_overlay received request")
    if not file.content_type or not file.content_type.startswith("image/"):
        logger.warning("Invalid file type: %s", file.content_type)
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Expected image (JPEG, PNG, etc.).",
        )

    try:
        img_bytes = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file: %s", e)
        raise HTTPException(status_code=400, detail="Failed to read image file.") from e

    if len(img_bytes) == 0:
        logger.warning("Empty image file received")
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        overlay_bytes = get_gradcam_overlay_bytes(img_bytes)
        logger.info("Grad-CAM overlay generated successfully")
        return Response(content=overlay_bytes, media_type="image/png")
    except RuntimeError as e:
        if "not loaded" in str(e):
            logger.error("Model not loaded: %s", e)
            raise HTTPException(
                status_code=503,
                detail="Model not ready. Ensure model and label map are loaded at startup.",
            ) from e
        logger.exception("RuntimeError in overlay: %s", e)
        raise
    except ValueError as e:
        logger.warning("ValueError in overlay: %s", e)
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Grad-CAM overlay failed: %s", e)
        raise HTTPException(status_code=500, detail="Overlay generation failed.") from e
