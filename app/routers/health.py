from fastapi import APIRouter, Response, status
from ..schemas.health import HealthResponse
from ..services.health_service import HealthService
import logging

router = APIRouter()
health_service = HealthService()
logger = logging.getLogger(__name__)


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="시스템 상태 확인",
    description="시스템의 전반적인 상태를 확인합니다. 메모리 사용량, 모델 상태, GPU 상태 등을 포함합니다.",
    tags=["health"],
)
async def health_check(response: Response):
    try:
        health_status = health_service.get_system_health()

        # 시스템 상태에 따른 HTTP 상태 코드 설정
        if health_status["status"] == "healthy":
            response.status_code = status.HTTP_200_OK
        else:
            response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

        return health_status
    except Exception as e:
        logger.error(f"Health check 처리 중 오류 발생: {str(e)}")
        response.status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        return {"status": "error", "error": str(e)}
