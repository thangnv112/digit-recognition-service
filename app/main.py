import time
import base64
from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import uvicorn

from app.config import get_settings
from app.utils.logger import setup_logging, get_logger
from app.schemas import PredictionResponse, DigitDetail, Base64PredictRequest, ErrorResponse, HealthResponse
from app.services.inference import DigitClassifier
from app.services.preprocessor import preprocess_image
from app import __version__

settings = get_settings()
setup_logging(settings.log_level)
logger = get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('request_count', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('request_latency_seconds', 'HTTP request latency', ['endpoint'])

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up application")
    classifier = DigitClassifier(settings.model_path)
    if not classifier.is_loaded():
        logger.warning("Classifier could not be loaded on startup", model_path=settings.model_path)
        app.state.classifier = None
    else:
        app.state.classifier = classifier
    yield
    logger.info("Shutting down application")

app = FastAPI(title="Handwritten Digit Recognition", version=__version__, lifespan=lifespan)

@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    start_time = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start_time
    
    endpoint = request.url.path
    if endpoint not in ['/metrics']:
        REQUEST_COUNT.labels(method=request.method, endpoint=endpoint, status=response.status_code).inc()
        REQUEST_LATENCY.labels(endpoint=endpoint).observe(duration)
        
    return response

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(success=False, error="Client Error", detail=exc.detail).model_dump()
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error("Unhandled exception", error=str(exc))
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(success=False, error="Internal Server Error", detail="An unexpected error occurred").model_dump()
    )

@app.get("/metrics")
async def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.get("/health", response_model=HealthResponse)
async def health():
    classifier = app.state.classifier
    model_loaded = classifier is not None and classifier.is_loaded()
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        model_loaded=model_loaded,
        version=__version__
    )

async def _process_prediction(image_bytes: bytes, classifier: DigitClassifier) -> PredictionResponse:
    if len(image_bytes) > settings.max_image_size_mb * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Image size exceeds maximum allowed")
        
    start_time = time.perf_counter()
    
    try:
        batch_tensor, boxes = preprocess_image(image_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
        
    predictions, confidences = classifier.predict(batch_tensor)
    
    details = []
    detected_text_parts = []
    
    for (pred, conf, box) in zip(predictions, confidences, boxes):
        details.append(DigitDetail(
            digit=pred,
            confidence=conf,
            box=list(box)
        ))
        if conf >= settings.confidence_threshold:
            detected_text_parts.append(str(pred))
        else:
            detected_text_parts.append("?")
            
    detected_text = "".join(detected_text_parts)
    inference_time_ms = (time.perf_counter() - start_time) * 1000
    logger.info("prediction_completed", detected_text=detected_text, inference_time_ms=f"{inference_time_ms:.2f}ms", count=len(details))
    
    return PredictionResponse(
        success=True,
        detected_text=detected_text,
        details=details,
        inference_time_ms=inference_time_ms
    )

@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    classifier = app.state.classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    image_bytes = await file.read()
    return await _process_prediction(image_bytes, classifier)

@app.post("/api/v1/predict/base64", response_model=PredictionResponse)
async def predict_base64(request: Base64PredictRequest):
    classifier = app.state.classifier
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
        
    try:
        image_bytes = base64.b64decode(request.image_base64)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 encoding")
        
    return await _process_prediction(image_bytes, classifier)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
