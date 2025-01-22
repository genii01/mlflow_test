import torch
from torch2trt import torch2trt, TRTModule
from torchvision.models import mobilenet_v3_large
import argparse
import os


def load_pth_model(model_path):
    """PTH 모델 파일 로드"""
    print(f"Loading PyTorch model from {model_path}")

    # 기본 모델 구조 생성
    model = mobilenet_v3_large(weights=None)

    # 저장된 가중치 로드
    checkpoint = torch.load(model_path, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def convert_to_tensorrt(model, input_shape=(1, 3, 224, 224), fp16_mode=True):
    """PyTorch 모델을 TensorRT로 변환"""
    print("Converting to TensorRT...")

    # 모델을 CUDA로 이동
    model = model.cuda()

    # 더미 입력 생성
    x = torch.ones(input_shape).cuda()

    # TensorRT 모델로 변환
    model_trt = torch2trt(model, [x], fp16_mode=fp16_mode)
    return model_trt


def save_tensorrt_model(model_trt, output_path):
    """TensorRT 모델 저장"""
    print(f"Saving TensorRT model to {output_path}")
    torch.save(model_trt.state_dict(), output_path)


def verify_tensorrt_model(model_path, input_shape=(1, 3, 224, 224)):
    """저장된 TensorRT 모델 검증"""
    print("Verifying saved TensorRT model...")

    model_trt = TRTModule()
    model_trt.load_state_dict(torch.load(model_path))

    # 테스트 입력으로 추론 시도
    x = torch.ones(input_shape).cuda()
    try:
        output = model_trt(x)
        print("Model verification successful!")
        return True
    except Exception as e:
        print(f"Model verification failed: {str(e)}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Convert PyTorch model to TensorRT")
    parser.add_argument(
        "--input_model",
        type=str,
        required=True,
        help="Path to input PyTorch model (.pth file)",
    )
    parser.add_argument(
        "--output_model", type=str, required=True, help="Path to save TensorRT model"
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="Batch size for TensorRT optimization"
    )
    parser.add_argument(
        "--fp16", action="store_true", help="Enable FP16 mode for faster inference"
    )

    args = parser.parse_args()

    # CUDA 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. TensorRT conversion requires CUDA.")
        return

    try:
        # 입력 모델 로드
        model = load_pth_model(args.input_model)

        # 입력 shape 설정
        input_shape = (args.batch_size, 3, 224, 224)

        # TensorRT 변환
        model_trt = convert_to_tensorrt(model, input_shape, args.fp16)

        # 모델 저장
        save_tensorrt_model(model_trt, args.output_model)

        # 저장된 모델 검증
        if verify_tensorrt_model(args.output_model, input_shape):
            print("Conversion completed successfully!")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    main()
