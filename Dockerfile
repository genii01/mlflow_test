# 베이스 이미지 선택
FROM python:3.10-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Poetry 설치
RUN pip install poetry

# Poetry 가상 환경 비활성화 (Docker 내부에서는 불필요)
RUN poetry config virtualenvs.create false

# 프로젝트 의존성 파일 복사
COPY pyproject.toml poetry.lock ./

# 의존성 설치
# --no-dev: 개발 의존성을 제외하고 프로덕션 의존성만 설치
# --no-interaction: 사용자 입력이 필요한 경우 기본값 사용
# --no-ansi: 컬러 출력과 스타일링을 비활성화
# --no-root: 프로젝트 자체를 설치하지 않고 의존성만 설치
RUN poetry install --no-interaction --no-ansi --no-root

# 소스 코드 복사
COPY . .

# 포트 설정
EXPOSE 8000

# 실행 명령어
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 