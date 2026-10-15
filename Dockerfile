# syntax=docker/dockerfile:1

# TORCH_VARIANT selects which torch build to install:
#   cpu   -> CPU-only wheels (default; works everywhere, much smaller image)
#   cu130 -> CUDA 13.0 wheels (needs an NVIDIA GPU + NVIDIA Container Toolkit)
ARG PYTHON_VERSION=3.11

# ---------------------------------------------------------------------------
# Builder: resolve the locked dependencies into a virtualenv
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS builder

ARG TORCH_VARIANT=cpu

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "poetry>=2,<3" poetry-plugin-export

WORKDIR /build
COPY pyproject.toml poetry.lock ./

# Export the lockfile, then pull torch out of it so the right build (CPU or
# CUDA) can be installed from PyTorch's own index. The CUDA runtime packages
# the lock pins for Linux are dropped too: the CUDA torch wheel brings its own,
# and the CPU build doesn't need them.
RUN poetry export --only main --without-hashes -f requirements.txt -o requirements.txt \
    && grep -E '^(torch|torchvision)==' requirements.txt | cut -d';' -f1 > torch.txt \
    && grep -vE '^(torch|torchvision|triton|nvidia-[a-z0-9-]+|cuda-[a-z0-9-]+)==' requirements.txt > deps.txt

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install -r torch.txt --index-url "https://download.pytorch.org/whl/${TORCH_VARIANT}" \
    && pip install --no-deps -r deps.txt

# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------
FROM python:${PYTHON_VERSION}-slim AS runtime

# libgl1/libglib2.0-0: required by opencv-python (pulled in by simple-lama-inpainting)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

# Model weights (Hugging Face + LaMa via torch.hub) download here on first use;
# mount a volume at /cache so they persist between runs.
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    HF_HOME=/cache/huggingface \
    TORCH_HOME=/cache/torch

RUN mkdir -p /cache /app/output && chown -R app:app /cache /app

WORKDIR /app
COPY --chown=app:app src ./src
COPY --chown=app:app web ./web

EXPOSE 8000

USER app

ENTRYPOINT ["python", "src/main.py"]
CMD ["--help"]
