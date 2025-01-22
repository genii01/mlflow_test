from fastapi import APIRouter, HTTPException
from ..schemas.dto_define import InferenceInput, InferenceResponse, PredictionResult
from ..services.inference_service import InferenceService
import logging

router = APIRouter()
inference_service = InferenceService()
logger = logging.getLogger(__name__)


@router.post("/predict", response_model=InferenceResponse)
async def predict(input_data: InferenceInput):
    try:
        # 이미지 전처리
        image_tensor = inference_service.preprocess_image(input_data.image_path)

        # 예측 수행
        predictions = inference_service.predict(image_tensor)

        # 응답 형식으로 변환
        response = InferenceResponse(
            predictions=[
                PredictionResult(category=cat, confidence=conf)
                for cat, conf in predictions
            ],
            artifact_uri="",
            run_id="",
        )
        return response

    except Exception as e:
        logger.error(f"예측 처리 중 오류 발생: {str(e)}")
        return InferenceResponse(
            predictions=[], error=str(e), artifact_uri="", run_id=""
        )
