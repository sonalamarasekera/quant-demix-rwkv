# syntax=docker/dockerfile:1
ARG DEVICE_TYPE=cuda
FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel && \
    python -m pip install --no-cache-dir poetry

COPY pyproject.toml ./
COPY poetry.lock ./
COPY src ./src

# Install PyTorch/torchaudio according to the build argument before Poetry
# so Poetry doesn't need to resolve GPU-specific wheels itself. Default is
# CUDA-enabled wheels; to force CPU use `--build-arg DEVICE_TYPE=cpu`.
RUN if [ "${DEVICE_TYPE}" = "cpu" ]; then \
            python -m pip install --no-cache-dir "torch" "torchaudio" -f https://download.pytorch.org/whl/cpu/torch_stable.html; \
        else \
            python -m pip install --no-cache-dir "torch" "torchaudio" -f https://download.pytorch.org/whl/cu118/torch_stable.html || true; \
        fi

# Install project deps via Poetry (torch already installed above)
RUN poetry install --without dev --no-interaction --no-ansi

COPY . ./

RUN groupadd --system app && useradd --system --gid app --create-home app
USER app

ENTRYPOINT ["python", "-m", "scripts.train"]
CMD ["--config", "configs/experiment/baseline_tf.yaml"]
