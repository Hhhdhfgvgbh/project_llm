FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md ./
COPY app ./app
COPY config ./config

RUN python -m pip install --upgrade pip && \
    python -m pip install .

EXPOSE 8501

CMD ["streamlit", "run", "app/ui.py", "--server.address=0.0.0.0", "--server.port=8501"]
