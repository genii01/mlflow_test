import mlflow
import os
from datetime import datetime
import logging
from typing import Dict, Any
import boto3
from botocore.client import Config

logger = logging.getLogger(__name__)


class MLflowService:
    def __init__(self):
        self.mlflow_tracking_uri = os.getenv(
            "MLFLOW_TRACKING_URI", "http://mlflow:5000"
        )
        self.experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "model-inference")
        self.s3_endpoint_url = os.getenv("MLFLOW_S3_ENDPOINT_URL", "http://minio:9000")
        self.aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID", "minio")
        self.aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY", "minio123")

        # S3 클라이언트 설정
        self.s3_client = boto3.client(
            "s3",
            endpoint_url=self.s3_endpoint_url,
            aws_access_key_id=self.aws_access_key_id,
            aws_secret_access_key=self.aws_secret_access_key,
            config=Config(signature_version="s3v4"),
            region_name="us-east-1",
        )

        # MLflow 설정
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        os.environ["MLFLOW_S3_ENDPOINT_URL"] = self.s3_endpoint_url

        # 실험 생성 또는 가져오기
        try:
            # 버킷 존재 여부 확인
            try:
                self.s3_client.head_bucket(Bucket="mlflow")
            except:
                self.s3_client.create_bucket(Bucket="mlflow")
                # 버킷 정책 설정
                bucket_policy = {
                    "Version": "2012-10-17",
                    "Statement": [
                        {
                            "Sid": "PublicRead",
                            "Effect": "Allow",
                            "Principal": "*",
                            "Action": ["s3:GetObject", "s3:ListBucket"],
                            "Resource": [
                                "arn:aws:s3:::mlflow",
                                "arn:aws:s3:::mlflow/*",
                            ],
                        }
                    ],
                }
                self.s3_client.put_bucket_policy(
                    Bucket="mlflow", Policy=str(bucket_policy)
                )

            # 먼저 실험이 존재하는지 확인
            experiment = mlflow.get_experiment_by_name(self.experiment_name)

            if experiment is None:
                # 실험이 없는 경우에만 새로 생성
                self.experiment_id = mlflow.create_experiment(
                    self.experiment_name, artifact_location="s3://mlflow/artifacts"
                )
            else:
                # 이미 존재하는 경우 해당 실험 ID 사용
                self.experiment_id = experiment.experiment_id

        except Exception as e:
            logger.error(f"MLflow 실험 생성 중 오류: {str(e)}")
            raise

    def log_prediction(
        self,
        image_path: str,
        predictions: list,
        model_version: str,
        metadata: Dict[str, Any] = None,
    ) -> str:
        """예측 결과를 MLflow에 기록"""
        try:
            with mlflow.start_run(experiment_id=self.experiment_id) as run:
                # 기본 태그 설정
                mlflow.set_tags(
                    {
                        "model_version": model_version,
                        "inference_timestamp": datetime.now().isoformat(),
                        "image_name": os.path.basename(image_path),
                    }
                )

                # 예측 결과 메트릭으로 기록
                for idx, (category, confidence) in enumerate(predictions):
                    mlflow.log_metric(f"confidence_top_{idx+1}", confidence)
                    mlflow.log_param(f"category_top_{idx+1}", category)

                # 추가 메타데이터 기록
                if metadata:
                    for key, value in metadata.items():
                        mlflow.log_param(key, value)

                # 아티팩트 저장 전에 버킷 접근 권한 확인
                try:
                    self.s3_client.head_object(
                        Bucket="mlflow",
                        Key=f"artifacts/{run.info.run_id}/input_images/{os.path.basename(image_path)}",
                    )
                except:
                    pass  # 객체가 없는 것은 정상

                # 이미지 파일을 아티팩트로 저장
                mlflow.log_artifact(image_path, "input_images")

                # 예측 결과 텍스트 파일 생성 및 저장
                result_path = (
                    f"/tmp/prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                )
                with open(result_path, "w") as f:
                    f.write(f"Prediction Results for {image_path}\n")
                    f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                    f.write(f"Model Version: {model_version}\n\n")
                    for idx, (category, confidence) in enumerate(predictions, 1):
                        f.write(f"{idx}. {category}: {confidence:.2f}%\n")

                mlflow.log_artifact(result_path, "prediction_results")
                os.remove(result_path)  # 임시 파일 삭제

                return run.info.run_id

        except Exception as e:
            logger.error(f"MLflow 로깅 중 오류 발생: {str(e)}")
            raise

    def get_run_info(self, run_id: str) -> Dict[str, Any]:
        """MLflow 실행 정보 조회"""
        try:
            run = mlflow.get_run(run_id)
            return {
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": run.info.start_time,
                "end_time": run.info.end_time,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
        except Exception as e:
            logger.error(f"MLflow 실행 정보 조회 중 오류 발생: {str(e)}")
            raise
