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
    Returns: pred_label, confidence, gradcam_overlay_b64 (base64 PNG).
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Expected image (JPEG, PNG, etc.).",
        )

    try:
        img_bytes = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail="Failed to read image file.") from e

    if len(img_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        result = predict_with_gradcam(img_bytes)
    except RuntimeError as e:
        if "not loaded" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Model not ready. Ensure model and label map are loaded at startup.",
            ) from e
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed.") from e

    return result


overlay_router = APIRouter(tags=["predict"])


@overlay_router.post("/predict_overlay")
async def predict_overlay(file: UploadFile = File(...)):
    """
    Classify brain MRI image and return the Grad-CAM overlay as PNG directly.
    """
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Expected image (JPEG, PNG, etc.).",
        )

    try:
        img_bytes = await file.read()
    except Exception as e:
        logger.exception("Failed to read uploaded file")
        raise HTTPException(status_code=400, detail="Failed to read image file.") from e

    if len(img_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty image file.")

    try:
        overlay_bytes = get_gradcam_overlay_bytes(img_bytes)
    except RuntimeError as e:
        if "not loaded" in str(e):
            raise HTTPException(
                status_code=503,
                detail="Model not ready. Ensure model and label map are loaded at startup.",
            ) from e
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.exception("Grad-CAM overlay failed")
        raise HTTPException(status_code=500, detail="Overlay generation failed.") from e

    return Response(content=overlay_bytes, media_type="image/png")
