# CPU-only validation image; CI builds this target without the large model stack.
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3 AS scaffold

WORKDIR /app

ENV UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN pip install --no-cache-dir uv==0.12.4

COPY pyproject.toml uv.lock ./
COPY src ./src
RUN uv sync --locked --no-dev

EXPOSE 8000
CMD ["uvicorn", "expl_drift_endpoint.app:app", "--host", "0.0.0.0", "--port", "8000"]

# The deployable endpoint installs the pinned Laya SDK and its CUDA PyTorch wheels.
FROM scaffold AS model-runtime
ENV USE_TF=0 \
    LAYA_MODEL_PATH=/repository \
    LAYA_MODEL_REVISION=7b928d828b7b0e022f929d9bd2e44165aa270148
RUN uv sync --locked --no-dev --extra model

LABEL org.opencontainers.image.title="expl_drift Laya score endpoint"
CMD ["uvicorn", "expl_drift_endpoint.app:model_app", "--host", "0.0.0.0", "--port", "8000"]
