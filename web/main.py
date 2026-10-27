"""
FastAPI app — YouTube comment spam detector.

Run: uvicorn main:app --reload --port 3019
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

_DEMO_ROOT = Path(__file__).resolve().parent.parent
if str(_DEMO_ROOT) not in sys.path:
    sys.path.insert(0, str(_DEMO_ROOT))

from spam_core import get_or_train_pipeline, sample_comments  # noqa: E402

_pipeline = None
_training = False
_train_error: str | None = None
_lock = threading.Lock()


def _train_in_background() -> None:
    global _pipeline, _training, _train_error
    with _lock:
        if _pipeline is not None or _training:
            return
        _training = True
        _train_error = None

    try:
        model = get_or_train_pipeline(compute_cv=False)
        with _lock:
            _pipeline = model
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _train_error = str(exc)
    finally:
        with _lock:
            _training = False


def _ensure_training_started() -> None:
    threading.Thread(target=_train_in_background, daemon=True).start()


def _status_payload() -> dict[str, Any]:
    with _lock:
        return {
            "ready": _pipeline is not None,
            "training": _training,
            "error": _train_error,
            "holdout_accuracy": _pipeline.holdout_accuracy if _pipeline else None,
            "cv_mean_accuracy": _pipeline.cv_mean_accuracy if _pipeline else None,
        }


app = FastAPI(title="Spam Detector")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(Path(__file__).parent / "static")), name="static")


@app.on_event("startup")
async def startup() -> None:
    _ensure_training_started()


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {"examples": sample_comments()},
    )


@app.get("/help", response_class=HTMLResponse)
async def help_page(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(request, "help.html")


@app.get("/api/status")
async def api_status() -> JSONResponse:
    if not _pipeline and not _training and _train_error is None:
        _ensure_training_started()
    return JSONResponse(_status_payload())


@app.post("/api/predict")
async def api_predict(body: PredictRequest) -> JSONResponse:
    with _lock:
        pipeline = _pipeline
        training = _training
        err = _train_error

    if pipeline is None:
        if err:
            raise HTTPException(status_code=503, detail=err)
        if training:
            raise HTTPException(
                status_code=503,
                detail="Model is still training. Try again shortly.",
            )
        _ensure_training_started()
        raise HTTPException(status_code=503, detail="Model is loading.")

    try:
        return JSONResponse(pipeline.predict(body.text))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
