FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 QUANTAI_DATA_DIR=/data
WORKDIR /app
COPY pyproject.toml README.md ./
COPY quantai ./quantai
COPY config ./config
RUN pip install --no-cache-dir . && useradd -r -u 10001 quantai && mkdir -p /data && chown quantai /data
USER quantai
EXPOSE 8000
CMD ["quantai", "serve", "--host", "0.0.0.0", "--port", "8000"]
