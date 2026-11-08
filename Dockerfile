FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml ./
COPY src/ ./src/

RUN python -m pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu \
    && python -m pip install --no-cache-dir .

COPY artifacts/byte_dqn_best.pt /models/byte_dqn_best.pt

ENV CACHE_POLICY=dqn-byte
ENV CACHE_CHECKPOINT=/models/byte_dqn_best.pt
ENV CACHE_CAPACITY_BYTES=24576
ENV CACHE_MAX_OBJECTS=16
ENV CACHE_DEVICE=cpu

EXPOSE 8000

CMD ["uvicorn", "smartcache.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
