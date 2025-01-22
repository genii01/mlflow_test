from pydantic import BaseModel
from typing import List, Optional


class InferenceInput(BaseModel):
    image_path: str


class PredictionResult(BaseModel):
    category: str
    confidence: float


class InferenceResponse(BaseModel):
    predictions: List[PredictionResult]
    error: Optional[str] = None
    artifact_uri: str
    run_id: Optional[str] = None
