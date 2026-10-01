from pydantic import BaseModel, Field


class DigitDetail(BaseModel):
    digit: int = Field(ge=0, le=9)
    confidence: float = Field(ge=0.0, le=1.0)
    box: list[int] = Field(min_length=4, max_length=4)

class PredictionResponse(BaseModel):
    success: bool = True
    detected_text: str
    details: list[DigitDetail] = Field(default_factory=list)
    inference_time_ms: float

class Base64PredictRequest(BaseModel):
    image_base64: str = Field(description="Base64 encoded image")

class ErrorResponse(BaseModel):
    success: bool = False
    error: str
    detail: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
