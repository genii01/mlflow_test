from pydantic import BaseModel
from typing import Dict, Any, Optional
from datetime import datetime


class MLflowRunResponse(BaseModel):
    run_id: str
    experiment_id: str
    status: str
    start_time: datetime
    end_time: Optional[datetime]
    metrics: Dict[str, float]
    params: Dict[str, str]
    tags: Dict[str, str]


class MLflowLogResponse(BaseModel):
    run_id: str
    artifact_uri: Optional[str] = None
    status: str
    message: str


class PredictionLogRequest(BaseModel):
    image_path: str
    predictions: list
    model_version: str
