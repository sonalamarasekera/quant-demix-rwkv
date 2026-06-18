# syntax=docker/dockerfile:1
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

RUN poetry install --without dev --no-interaction --no-ansi

COPY . ./

RUN groupadd --system app && useradd --system --gid app --create-home app
USER app

ENTRYPOINT ["python", "-m", "scripts.train"]
CMD ["--config", "configs/experiment/baseline_tf.yaml"]
