import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import matplotlib.pyplot as plt
import requests
import os


def load_model():
    # MobileNetV3 모델 로드 (사전 학습된 가중치 사용)
    model = mobilenet_v3_large(weights=MobileNet_V3_Large_Weights.DEFAULT)
    model.eval()  # 평가 모드로 설정
    return model


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

    # RGBA나 다른 채널 이미지를 RGB로 변환
    if image.mode != "RGB":
        image = image.convert("RGB")

    # 원본 이미지 저장
    original_image = image.copy()
    # 전처리된 이미지 텐서
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_image


def predict_and_visualize(model, image_tensor, original_image, image_path):
    # ImageNet 클래스 레이블 로드
    labels = download_labels_if_needed()

    # 추론
    with torch.no_grad():
        outputs = model(image_tensor)

    # 상위 5개 예측 결과 얻기
    _, indices = torch.sort(outputs, descending=True)
    percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

    # 결과 출력 및 시각화
    plt.figure(figsize=(12, 6))

    # 원본 이미지 표시
    plt.subplot(1, 2, 1)
    plt.imshow(original_image)
    plt.title("Input Image")
    plt.axis("off")

    # 예측 결과 막대 그래프
    plt.subplot(1, 2, 2)
    top_5_indices = indices[0][:5].tolist()
    top_5_categories = [
        labels[idx] if idx < len(labels) else f"Class {idx}" for idx in top_5_indices
    ]
    top_5_percentages = [percentages[idx].item() for idx in top_5_indices]

    # 확률값이 높은 순서대로 정렬
    sorted_pairs = sorted(
        zip(top_5_categories, top_5_percentages), key=lambda x: x[1], reverse=True
    )
    top_5_categories, top_5_percentages = zip(*sorted_pairs)

    # 막대 그래프 생성 (위에서부터 아래로)
    y_pos = range(len(top_5_categories))[::-1]  # 역순으로 위치 설정
    bars = plt.barh(y_pos, top_5_percentages)
    plt.yticks(y_pos, top_5_categories)
    plt.xlabel("Confidence (%)")
    plt.title("Top 5 Predictions")

    # 퍼센트 값 표시
    for i, bar in enumerate(bars):
        plt.text(
            bar.get_width(),
            bar.get_y() + bar.get_height() / 2,
            f"{top_5_percentages[i]:.1f}%",
            va="center",
        )

    plt.tight_layout()

    # 결과 이미지 저장
    result_path = f'result_{image_path.split("/")[-1]}'
    plt.savefig(result_path)
    print(f"결과가 {result_path}에 저장되었습니다.")

    # 콘솔에 결과 출력
    print("\n예측 결과:")
    for cat, perc in zip(top_5_categories, top_5_percentages):
        print(f"{cat}: {perc:.2f}%")


def save_model(model, path="mobilenetv3_model.pth"):
    # 모델 저장
    torch.save(
        {"model_state_dict": model.state_dict(), "model_name": "mobilenet_v3_large"},
        path,
    )
    print(f"모델이 {path}에 저장되었습니다.")


def load_saved_model(path="mobilenetv3_model.pth"):
    # 저장된 모델 불러오기
    checkpoint = torch.load(path, weights_only=True)  # weights_only=True 추가
    model = mobilenet_v3_large(weights=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def download_labels_if_needed():
    labels_file = "imagenet_classes.txt"

    # 레이블 파일이 없는 경우에만 다운로드
    if not os.path.exists(labels_file):
        response = requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        with open(labels_file, "w") as f:
            f.write(response.text)

    # 레이블 파일 읽기
    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    return labels


def main():
    # 테스트할 이미지 경로들
    test_images = [
        "truck.png",  # 여기에 실제 테스트할 이미지 경로를 입력하세요
        # 추가 테스트 이미지들...
    ]

    # 모델 로드
    model = load_model()

    # 모델 저장
    save_model(model)

    # 저장된 모델 다시 불러오기
    loaded_model = load_saved_model()

    # 각 테스트 이미지에 대해 예측 수행
    for image_path in test_images:
        try:
            print(f"\n처리중인 이미지: {image_path}")
            image_tensor, original_image = preprocess_image(image_path)
            predict_and_visualize(
                loaded_model, image_tensor, original_image, image_path
            )
        except Exception as e:
            print(f"이미지 {image_path} 처리 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
