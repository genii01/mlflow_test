from pydantic import BaseModel, ConfigDict
from typing import Dict, Optional, Union, Any
from datetime import datetime


class HealthResponse(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    status: str
    version: str
    timestamp: datetime
    uptime: float
    model_status: str
    memory_usage: Dict[str, float]
    gpu_status: Optional[Dict[str, Dict[str, str]]] = None
