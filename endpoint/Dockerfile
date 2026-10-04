# CPU scaffold base, pinned to the image resolved during local qualification.
FROM python:3.12-slim-bookworm@sha256:54c85f3c47607a77f32adec749d3c81d1348bf25833671f512b26a9b6d778cb3

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

LABEL org.opencontainers.image.title="expl_drift extraction endpoint scaffold"

EXPOSE 8000

CMD ["uvicorn", "expl_drift_endpoint.app:app", "--host", "0.0.0.0", "--port", "8000"]
