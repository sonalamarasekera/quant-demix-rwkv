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
ENV POETRY_VERSION=1.8.3 \
    POETRY_HOME="/opt/poetry" \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=true
    
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="$POETRY_HOME/bin:$PATH"

# CRITICAL FOR CACHING: Copy ONLY configuration files first
COPY pyproject.toml poetry.lock ./

# Modern approach to PyTorch: Inject explicit source tracking to bypass local wheel build limits
ARG DEVICE_TYPE
RUN if [ "${DEVICE_TYPE}" = "cpu" ]; then \
        poetry source add --priority=explicit pytorch https://pytorch.org/whl/cpu; \
    else \
        poetry source add --priority=explicit pytorch https://pytorch.org/whl/cu121; \
    fi && \
    poetry add --source pytorch torch torchaudio

# Install all project dependencies into an isolated local .venv folder
RUN poetry install --without dev --no-root --no-ansi


# ==========================================
# STAGE 2: The Production Runner (Ultra-Lean & Secure)
# ==========================================
FROM python:3.10-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Install ONLY the bare runtime system libraries required to run PyTorch and Audio tools
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Securely copy over ONLY the compiled virtual environment from the builder
COPY --from=builder /build/.venv /app/.venv

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
