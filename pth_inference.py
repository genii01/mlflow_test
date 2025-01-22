import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v3_large
import matplotlib.pyplot as plt
import requests
import os


def load_saved_model(path="mobilenetv3_model.pth"):
    checkpoint = torch.load(path, weights_only=True)
    model = mobilenet_v3_large(weights=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_image(image_path):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path)
    if image.mode != "RGB":
        image = image.convert("RGB")

    original_image = image.copy()
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor, original_image


def download_labels_if_needed():
    labels_file = "imagenet_classes.txt"
    if not os.path.exists(labels_file):
        response = requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        with open(labels_file, "w") as f:
            f.write(response.text)

    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def inference(model, image_path):
    # 이미지 전처리
    image_tensor, original_image = preprocess_image(image_path)

    # 레이블 로드
    labels = download_labels_if_needed()

    # 추론
    with torch.no_grad():
        outputs = model(image_tensor)

    # 상위 5개 예측 결과 얻기
    _, indices = torch.sort(outputs, descending=True)
    percentages = torch.nn.functional.softmax(outputs, dim=1)[0] * 100

    # 상위 5개 결과 추출
    top_5_indices = indices[0][:5].tolist()
    top_5_categories = [labels[idx] for idx in top_5_indices]
    top_5_percentages = [percentages[idx].item() for idx in top_5_indices]

    # 확률값이 높은 순서대로 정렬
    sorted_pairs = sorted(
        zip(top_5_categories, top_5_percentages), key=lambda x: x[1], reverse=True
    )
    top_5_categories, top_5_percentages = zip(*sorted_pairs)

    # 콘솔에 결과 출력
    print("\n예측 결과:")
    for cat, perc in zip(top_5_categories, top_5_percentages):
        print(f"{cat}: {perc:.2f}%")


if __name__ == "__main__":
    # 모델 파일 경로
    model_path = "mobilenetv3_model.pth"

    # 추론할 이미지 경로
    image_path = "truck.png"  # 여기에 실제 이미지 경로를 입력하세요

    try:
        # 모델 로드
        model = load_saved_model(model_path)

        # 추론 실행
        inference(model, image_path)

    except Exception as e:
        print(f"오류 발생: {str(e)}")
