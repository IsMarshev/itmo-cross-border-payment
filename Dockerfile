# Демо-стенды сигнального слоя одним контейнером.
#
#   docker build -t signal-layer-demo .
#   docker run --rm -p 8100:8100 signal-layer-demo
#
# Симуляция — http://localhost:8100, статический стенд — http://localhost:8100/stand/
#
# Слой считает хронологический walk-forward один раз при старте (~3 с на пять
# коридоров), поэтому первый запрос стоит подождать: контейнер поднимается
# готовым, а не прогревается на первом клике.

FROM python:3.12-slim

# uv ставит зависимости по uv.lock: тот же набор версий, что и локально.
RUN pip install --no-cache-dir uv==0.12.4

WORKDIR /app

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/app/.venv \
    PYTHONUNBUFFERED=1

# Сначала зависимости — слой кэшируется и не пересобирается при правке демо.
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN uv sync --frozen --no-dev

COPY currency_data ./currency_data
COPY demo ./demo

ENV PATH="/app/.venv/bin:$PATH" \
    SIGNAL_LAYER_DATA_DIR=/app/currency_data \
    SIM_HOST=0.0.0.0 \
    SIM_PORT=8100

EXPOSE 8100

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8100/api/corridors').read()"

CMD ["python", "demo/sim/server.py"]
