from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import router_svc, health, mlflow
import logging

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

app = FastAPI(
    title="Image Classification API",
    description="MobileNetV3 ONNX 모델을 사용한 이미지 분류 API",
    version="1.0.0",
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 실제 운영 환경에서는 구체적인 도메인을 지정하세요
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(router_svc.router, prefix="/api/v1", tags=["inference"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])
app.include_router(mlflow.router, prefix="/api/v1/mlflow", tags=["mlflow"])


@app.get("/")
async def root():
    return {"message": "Image Classification API is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
