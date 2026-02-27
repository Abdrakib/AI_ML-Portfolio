# Brain Tumor Classification API

FastAPI backend for brain MRI tumor classification with Grad-CAM visualization.

## Setup

1. **Place model artifacts** (from notebook training):
   - `models/v1/adv_final_tfhub.keras` – trained model
   - `models/v1/label_to_idx.json` – label mapping

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the API**:
   ```bash
   uvicorn app.main:app --reload
   ```

## Endpoints

- `GET /health` – Liveness
- `GET /ready` – Readiness (model loaded)
- `POST /predict` – Upload image, get prediction + Grad-CAM overlay (JSON with base64)
- `POST /predict_overlay` – Upload image, get Grad-CAM overlay as PNG directly

## POST /predict

**Request**: `multipart/form-data` with `file` (image: JPEG, PNG, etc.)

**Response**:
```json
{
  "pred_label": "yes",
  "confidence": 0.9234,
  "probs": {"no": 0.0766, "yes": 0.9234},
  "gradcam_overlay_b64": "<base64-encoded PNG>"
}
```

Decode `gradcam_overlay_b64` to display the Grad-CAM heatmap overlay image.
