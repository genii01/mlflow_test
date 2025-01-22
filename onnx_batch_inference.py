import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def preprocess_images(image_paths, batch_size=4):
    # 이미지 전처리를 위한 변환 정의
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image_tensors = []
    original_images = []

    for path in image_paths:
        # 이미지 로드 및 전처리
        image = Image.open(path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        # 원본 이미지 저장
        original_images.append(image.copy())

        # 이미지 전처리
        image_tensor = transform(image).unsqueeze(0)
        image_tensors.append(image_tensor)

    # 배치로 결합
    batch_tensor = np.concatenate(image_tensors, axis=0)
    return batch_tensor, original_images


def load_labels():
    with open("imagenet_classes.txt", "r") as f:
        return [line.strip() for line in f.readlines()]


def predict_and_visualize_batch(session, batch_tensor, original_images, image_paths):
    # 레이블 로드
    labels = load_labels()

    # 배치 추론 실행
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: batch_tensor})
    scores = outputs[0]

    # softmax 적용
    scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    scores = scores * 100  # 퍼센트로 변환

    # 각 이미지별 예측 결과 출력
    for idx, image_path in enumerate(image_paths):
        # 상위 5개 예측 결과 얻기
        top_5_indices = np.argsort(scores[idx])[-5:][::-1]
        top_5_categories = [labels[i] for i in top_5_indices]
        top_5_percentages = [scores[idx][i] for i in top_5_indices]

        # 콘솔에 결과 출력
        print(f"\n이미지 {image_path}의 예측 결과:")
        for cat, perc in zip(top_5_categories, top_5_percentages):
            print(f"{cat}: {perc:.2f}%")


def main():
    # ONNX 모델 로드
    model_path = "mobilenetv3.onnx"
    session = onnxruntime.InferenceSession(model_path)

    # 테스트할 이미지들
    test_images = [
        "truck.png",
        "truck.png",
        "truck.png",
        "truck.png",
        # 추가 테스트 이미지들...
    ]

    # 배치 크기
    batch_size = 4

    # 이미지를 배치 크기만큼 나누어 처리
    for i in range(0, len(test_images), batch_size):
        batch_images = test_images[i : i + batch_size]
        try:
            print(f"\n배치 처리 중: {batch_images}")
            batch_tensor, original_images = preprocess_images(batch_images)
            predict_and_visualize_batch(
                session, batch_tensor, original_images, batch_images
            )
        except Exception as e:
            print(f"배치 처리 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
