from fastapi import APIRouter, HTTPException
from ..services.mlflow_service import MLflowService
from ..schemas.mlflow import MLflowRunResponse, MLflowLogResponse, PredictionLogRequest
import logging
import mlflow

router = APIRouter()
mlflow_service = MLflowService()
logger = logging.getLogger(__name__)


@router.get("/runs/{run_id}", response_model=MLflowRunResponse)
async def get_run(run_id: str):
    """MLflow 실행 정보 조회"""
    try:
        run_info = mlflow_service.get_run_info(run_id)
        return run_info
    except Exception as e:
        logger.error(f"MLflow 실행 정보 조회 중 오류: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/log-prediction", response_model=MLflowLogResponse)
async def log_prediction(request: PredictionLogRequest):
    """예측 결과를 MLflow에 기록"""
    try:
        run_id = mlflow_service.log_prediction(
            image_path=request.image_path,
            predictions=request.predictions,
            model_version=request.model_version,
        )
        # MLflow 실행 정보를 가져와서 artifact_uri 포함
        run = mlflow.get_run(run_id)
        return MLflowLogResponse(
            run_id=run_id,
            artifact_uri=run.info.artifact_uri,  # artifact_uri 추가
            status="success",
            message="Successfully logged prediction to MLflow",
        )
    except Exception as e:
        error_message = str(e)
        if "NoSuchBucket" in error_message:
            error_message = "MLflow 아티팩트 저장소(S3 버킷)가 존재하지 않습니다. 시스템 관리자에게 문의하세요."
        logger.error(f"MLflow 로깅 중 오류: {error_message}")
        raise HTTPException(status_code=500, detail=error_message)
