import psutil
import time
import torch
import onnxruntime
from datetime import datetime
from typing import Dict, Optional
import logging

logger = logging.getLogger(__name__)


class HealthService:
    def __init__(self):
        self.start_time = time.time()
        self.version = "1.0.0"

    def get_system_health(self) -> Dict:
        try:
            # 메모리 사용량 확인
            memory = psutil.virtual_memory()
            memory_usage = {
                "total": round(memory.total / (1024**3), 2),  # GB
                "available": round(memory.available / (1024**3), 2),  # GB
                "percent": memory.percent,
            }

            # GPU 상태 확인 (CUDA 사용 가능한 경우)
            gpu_status = self._check_gpu_status()

            # ONNX 모델 상태 확인
            model_status = self._check_model_status()

            return {
                "status": "healthy",
                "version": self.version,
                "timestamp": datetime.now(),
                "uptime": round(time.time() - self.start_time, 2),
                "model_status": model_status,
                "memory_usage": memory_usage,
                "gpu_status": gpu_status,
            }
        except Exception as e:
            logger.error(f"Health check 중 오류 발생: {str(e)}")
            return {
                "status": "unhealthy",
                "version": self.version,
                "timestamp": datetime.now(),
                "error": str(e),
            }

    def _check_gpu_status(self) -> Optional[Dict]:
        try:
            if torch.cuda.is_available():
                gpu_status = {}
                for i in range(torch.cuda.device_count()):
                    gpu = torch.cuda.get_device_properties(i)
                    memory = torch.cuda.memory_stats(i)
                    gpu_status[f"gpu_{i}"] = {
                        "name": gpu.name,
                        "total_memory": f"{gpu.total_memory / (1024**2):.2f} MB",
                        "memory_allocated": f"{torch.cuda.memory_allocated(i) / (1024**2):.2f} MB",
                        "memory_reserved": f"{torch.cuda.memory_reserved(i) / (1024**2):.2f} MB",
                    }
                return gpu_status
            return None
        except Exception as e:
            logger.warning(f"GPU 상태 확인 중 오류: {str(e)}")
            return None

    def _check_model_status(self) -> str:
        try:
            # ONNX 모델 파일 존재 여부 및 세션 생성 가능 여부 확인
            model_path = "mobilenetv3.onnx"
            onnxruntime.InferenceSession(model_path)
            return "ready"
        except Exception as e:
            logger.error(f"모델 상태 확인 중 오류: {str(e)}")
            return "unavailable"
