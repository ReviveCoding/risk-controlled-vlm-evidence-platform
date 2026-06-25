FROM python:3.13-slim AS builder
WORKDIR /build
COPY pyproject.toml README.md ./
COPY src ./src
RUN python -m pip install --no-cache-dir --upgrade pip build && python -m build --wheel

FROM python:3.13-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CONTROL_EVIDENCE_STATE_DIR=/app/state \
    CONTROL_EVIDENCE_MAX_WORKERS=2 \
    CONTROL_EVIDENCE_MAX_QUEUE=4 \
    CONTROL_EVIDENCE_TIMEOUT_SECONDS=5 \
    CONTROL_EVIDENCE_MAX_REQUEST_BYTES=1000000 \
    CONTROL_EVIDENCE_IDEMPOTENCY_RETENTION_SECONDS=2592000 \
    CONTROL_EVIDENCE_IDEMPOTENCY_MAX_RECORDS=100000
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/state \
    && chown -R appuser:appuser /app/state
WORKDIR /app
COPY --from=builder /build/dist/*.whl /tmp/package.whl
RUN python -m pip install --no-cache-dir /tmp/package.whl "fastapi>=0.137,<1" "uvicorn>=0.30,<1" \
    && rm /tmp/package.whl
USER appuser
VOLUME ["/app/state"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/ready', timeout=2)"
CMD ["uvicorn", "control_evidence.api:create_app_from_env", "--factory", "--host", "0.0.0.0", "--port", "8000"]
