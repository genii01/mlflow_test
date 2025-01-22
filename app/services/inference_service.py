import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from typing import List, Tuple, Dict, Any
import logging
import datetime
import os

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class InferenceService:
    def __init__(self, model_path: str = "mobilenetv3.onnx"):
        self.session = onnxruntime.InferenceSession(model_path)
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.labels = self._load_labels()

    def _load_labels(self) -> List[str]:
        with open("imagenet_classes.txt", "r") as f:
            return [line.strip() for line in f.readlines()]

    def preprocess_image(self, image_path: str) -> np.ndarray:
        try:
            image = Image.open(image_path)
            if image.mode != "RGB":
                image = image.convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0).numpy()
            return image_tensor
        except Exception as e:
            logger.error(f"이미지 전처리 중 오류 발생: {str(e)}")
            raise

    def predict(self, image_tensor: np.ndarray) -> List[Tuple[str, float]]:
        try:
            # 추론 실행
            input_name = self.session.get_inputs()[0].name
            outputs = self.session.run(None, {input_name: image_tensor})
            scores = outputs[0]

            # softmax 적용
            scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
            scores = scores * 100  # 퍼센트로 변환

            # 상위 5개 예측 결과 얻기
            top_5_indices = np.argsort(scores[0])[-5:][::-1]
            predictions = [
                (self.labels[idx], float(scores[0][idx])) for idx in top_5_indices
            ]

            # Create timestamp for unique filename
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

            # Create prediction result string
            result_str = f"Timestamp: {timestamp}\n"
            for label, confidence in predictions:
                result_str += f"Class: {label}, Confidence: {confidence:.2f}%\n"

            # Save prediction result to file in /app/models directory
            os.makedirs("/app/models", exist_ok=True)
            filename = f"/app/models/prediction_{timestamp}.txt"
            with open(filename, "w") as f:
                f.write(result_str)

            return predictions
        except Exception as e:
            logger.error(f"추론 중 오류 발생: {str(e)}")
            raise
