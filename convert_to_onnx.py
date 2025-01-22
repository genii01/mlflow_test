import torch
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
import onnx
import onnxruntime


def load_pytorch_model(model_path="mobilenetv3_model.pth"):
    # 저장된 모델 파일 로드
    checkpoint = torch.load(model_path, weights_only=True)  # weights_only=True 추가
    model = mobilenet_v3_large(weights=None)  # 기본 모델 구조 생성

    # state_dict 키 확인 및 로드
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def convert_to_onnx(model, save_path="mobilenetv3.onnx"):
    # 더미 입력 생성 (배치 크기 1, 채널 3, 높이 224, 너비 224)
    dummy_input = torch.randn(1, 3, 224, 224)

    # ONNX 변환
    torch.onnx.export(
        model,  # PyTorch 모델
        dummy_input,  # 모델 입력값 예시
        save_path,  # 저장할 경로
        export_params=True,  # 모델 파라미터 저장
        opset_version=11,  # ONNX 버전
        do_constant_folding=True,  # 최적화 옵션
        input_names=["input"],  # 입력 이름
        output_names=["output"],  # 출력 이름
        dynamic_axes={
            "input": {0: "batch_size"},  # 가변적인 배치 크기
            "output": {0: "batch_size"},
        },
    )

    # ONNX 모델 검증
    onnx_model = onnx.load(save_path)
    onnx.checker.check_model(onnx_model)
    print(f"ONNX 모델이 {save_path}에 저장되었습니다.")

    # ONNX Runtime으로 테스트
    ort_session = onnxruntime.InferenceSession(save_path)
    outputs = ort_session.run(None, {"input": dummy_input.numpy()})
    print("ONNX 모델 변환 및 테스트 완료!")


def main():
    # PyTorch 모델 로드 (경로 지정)
    pytorch_model = load_pytorch_model("./mobilenetv3_model.pth")

    # ONNX 변환
    convert_to_onnx(pytorch_model)


if __name__ == "__main__":
    main()
