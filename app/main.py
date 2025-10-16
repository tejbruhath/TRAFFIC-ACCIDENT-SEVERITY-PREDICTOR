from typing import Any, Dict

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import RootModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

app = FastAPI(title="Traffic Accident Severity API")

# Load model at startup
try:
    _MODEL_PATH = "model/rf_v1.joblib"
    model = joblib.load(_MODEL_PATH)
    FEATURE_NAMES = getattr(model, "feature_names_in_", None)
except Exception:
    model = None
    FEATURE_NAMES = None


class PredictPayload(RootModel[Dict[str, Any]]):
    def to_dict(self) -> Dict[str, Any]:
        return dict(self.root)


# Static files and index
app.mount("/static", StaticFiles(directory="app/static"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("app/static/index.html")


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_path": _MODEL_PATH,
        "n_features": int(FEATURE_NAMES.size) if FEATURE_NAMES is not None else None,
    }


@app.post("/predict")
def predict(payload: PredictPayload) -> Dict[str, Any]:
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")

    record = payload.to_dict()
    X = pd.DataFrame([record])
    X_enc = pd.get_dummies(X, dummy_na=True)
    if FEATURE_NAMES is not None:
        X_enc = X_enc.reindex(columns=FEATURE_NAMES, fill_value=0)

    try:
        pred = model.predict(X_enc)[0]
        proba = float(max(model.predict_proba(X_enc)[0])) if hasattr(model, "predict_proba") else None
    except Exception as err:
        raise HTTPException(status_code=400, detail=f"Prediction failed: {err}")

    return {
        "prediction": str(pred),
        "confidence": proba,
        "features_used": FEATURE_NAMES.tolist() if FEATURE_NAMES is not None else list(X_enc.columns),
    }
