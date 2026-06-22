# syntax=docker/dockerfile:1
ARG DEVICE_TYPE=cuda

# ==========================================
# STAGE 1: The Builder (Compiles and Resolves Everything)
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /build

# Install compilation tools safely in the isolated build stage
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry pinned to a modern production release
ENV POETRY_VERSION=2.4.1 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true
    
RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="$POETRY_HOME/bin:$PATH"

# CRITICAL FOR CACHING: Copy ONLY configuration files first
COPY pyproject.toml poetry.lock ./

# Modern approach to PyTorch: Inject explicit source tracking to bypass local wheel build limits
# ARG DEVICE_TYPE
# RUN if [ "${DEVICE_TYPE}" = "cpu" ]; then \
#         poetry source add --priority=explicit pytorch https://pytorch.org/whl/cpu; \
#     else \
#         poetry source add --priority=explicit pytorch https://pytorch.org/whl/cu121; \
#     fi && \
#     poetry add --source pytorch torch torchaudio

# Install all project dependencies into an isolated local .venv folder
ARG WITH_CUDA=true
RUN if [ "${WITH_CUDA}" = "true" ]; then \
        poetry install --only main --with cuda --no-root --no-ansi; \
    else \
        poetry install --only main --no-root --no-ansi; \
    fi

# ==========================================
# STAGE 2: Interactive Development / Training Target
# ==========================================
FROM python:3.10-slim AS dev

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/.venv /app/.venv

# ==========================================
# STAGE 3: The Production Runner
# ==========================================

FROM dev AS runner

ENV PYTHONPATH="/app/src"

# Copy your actual codebase last. Source updates now rebuild in under 2 seconds!
COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY pyproject.toml ./ 

# Enforce secure least-privilege access execution
RUN groupadd --system app && useradd --system --gid app --create-home app
USER app

ENTRYPOINT ["python", "-m", "scripts.train"]
CMD ["--config", "configs/experiment/baseline_tf.yaml"]