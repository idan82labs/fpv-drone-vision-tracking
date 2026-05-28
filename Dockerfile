FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    DATA_DIR=/data

WORKDIR /app

RUN pip install --no-cache-dir opencv-python-headless==4.10.0.84 numpy==2.1.3

COPY scripts/tube_labeling_server.py /app/scripts/tube_labeling_server.py
COPY web/tube_labeler /app/web/tube_labeler
COPY deploy_assets /seed
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 8080

CMD ["/app/docker-entrypoint.sh"]
