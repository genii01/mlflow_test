import onnxruntime
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


def preprocess_image(image_path):
    # 이미지 전처리를 위한 변환 정의
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # 이미지 로드 및 전처리
    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 원본 이미지 저장
    original_image = image.copy()

    # 이미지 전처리
    image_tensor = transform(image).unsqueeze(0).numpy()
    return image_tensor, original_image


def load_labels():
    with open("imagenet_classes.txt", "r") as f:
        return [line.strip() for line in f.readlines()]


def predict_and_visualize(session, image_tensor, original_image, image_path):
    # 레이블 로드
    labels = load_labels()

    # 추론 실행
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: image_tensor})
    scores = outputs[0]

    # softmax 적용
    scores = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    scores = scores * 100  # 퍼센트로 변환

    # 상위 5개 예측 결과 얻기
    top_5_indices = np.argsort(scores[0])[-5:][::-1]
    top_5_categories = [labels[idx] for idx in top_5_indices]
    top_5_percentages = [scores[0][idx] for idx in top_5_indices]

    # 콘솔에 결과 출력
    print("\n예측 결과:")
    for cat, perc in zip(top_5_categories, top_5_percentages):
        print(f"{cat}: {perc:.2f}%")


def main():
    # ONNX 모델 로드
    model_path = "mobilenetv3.onnx"
    session = onnxruntime.InferenceSession(model_path)

    # 테스트 이미지
    test_images = [
        "truck.png",
        "truck.png",
        "truck.png",
        "truck.png",  # 테스트할 이미지 경로
        # 추가 테스트 이미지들...
    ]

    # 각 테스트 이미지에 대해 예측 수행
    for image_path in test_images:
        from time import sleep

        sleep(1)
        try:
            print(f"\n처리중인 이미지: {image_path}")
            image_tensor, original_image = preprocess_image(image_path)
            predict_and_visualize(session, image_tensor, original_image, image_path)
        except Exception as e:
            print(f"이미지 {image_path} 처리 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
