import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision.models import mobilenet_v3_large
import os
from pathlib import Path


def load_saved_model(path="mobilenetv3_fullmodel.pth"):
    checkpoint = torch.load(path, weights_only=True)
    model = mobilenet_v3_large(weights=None)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def preprocess_images(image_paths):
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    batch_tensors = []
    original_images = []

    for img_path in image_paths:
        image = Image.open(img_path)
        if image.mode != "RGB":
            image = image.convert("RGB")

        original_images.append(image.copy())
        image_tensor = transform(image)
        batch_tensors.append(image_tensor)

    # 배치로 스택
    batch = torch.stack(batch_tensors)
    return batch, original_images


def download_labels_if_needed():
    labels_file = "imagenet_classes.txt"
    if not os.path.exists(labels_file):
        import requests

        response = requests.get(
            "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        )
        with open(labels_file, "w") as f:
            f.write(response.text)

    with open(labels_file, "r") as f:
        labels = [line.strip() for line in f.readlines()]
    return labels


def predict_batch(image_paths, model_path="mobilenetv3_fullmodel.pth"):
    try:
        # 모델 로드
        model = load_saved_model(model_path)

        # 이미지 배치 전처리
        image_batch, original_images = preprocess_images(image_paths)

        # ImageNet 클래스 레이블 로드
        labels = download_labels_if_needed()

        # 배치 추론
        with torch.no_grad():
            outputs = model(image_batch)

        # 각 이미지별 결과 처리
        for idx, output in enumerate(outputs):
            print(f"\n이미지 {image_paths[idx]} 예측 결과:")

            # 상위 5개 예측 결과 얻기
            percentages = torch.nn.functional.softmax(output, dim=0) * 100
            _, indices = torch.sort(output, descending=True)

            # 상위 5개 결과 추출
            top_5_indices = indices[:5].tolist()
            top_5_categories = [labels[idx] for idx in top_5_indices]
            top_5_percentages = [percentages[idx].item() for idx in top_5_indices]

            # 결과 출력
            for cat, perc in zip(top_5_categories, top_5_percentages):
                print(f"{cat}: {perc:.2f}%")

    except Exception as e:
        print(f"에러 발생: {str(e)}")


if __name__ == "__main__":
    # 예측할 이미지들의 경로 리스트 (배치 크기: 4)
    image_paths = [
        "truck.png",
        "truck.png",
        "truck.png",
        "truck.png",
    ]

    model_path = "mobilenetv3_model.pth"

    # 배치 추론 실행
    predict_batch(image_paths, model_path)
