#!/bin/sh
set -eu

DATA_DIR="${DATA_DIR:-/data}"
PACKET_NAME="${PACKET_NAME:-tube_hard_negative_review_packet_thr060_top8}"
CSV_DIR="$DATA_DIR/$PACKET_NAME"
VIDEO_DIR="$DATA_DIR/videos"

mkdir -p "$DATA_DIR"

if [ ! -f "$CSV_DIR/tube_alternatives_to_label.csv" ]; then
  if [ ! -f "/seed/$PACKET_NAME/tube_alternatives_to_label.csv" ]; then
    echo "Missing label packet: $CSV_DIR/tube_alternatives_to_label.csv" >&2
    echo "Seed a Fly volume under $DATA_DIR or build with local deploy_assets/$PACKET_NAME." >&2
    exit 2
  fi
  rm -rf "$CSV_DIR"
  cp -a "/seed/$PACKET_NAME" "$CSV_DIR"
fi

mkdir -p /app/results
ln -sfn "$CSV_DIR" "/app/results/$PACKET_NAME"

if [ ! -d "$VIDEO_DIR" ] || [ -z "$(find "$VIDEO_DIR" -maxdepth 1 -name '*.MP4' -print -quit 2>/dev/null)" ]; then
  if [ ! -d /seed/videos ] || [ -z "$(find /seed/videos -maxdepth 1 -name '*.MP4' -print -quit 2>/dev/null)" ]; then
    echo "Missing review videos under $VIDEO_DIR and /seed/videos." >&2
    echo "Seed a Fly volume under $DATA_DIR/videos or build with local deploy_assets/videos." >&2
    exit 2
  fi
  rm -rf "$VIDEO_DIR"
  cp -a /seed/videos "$VIDEO_DIR"
fi

exec python /app/scripts/tube_labeling_server.py \
  --host 0.0.0.0 \
  --port "${PORT:-8080}" \
  --csv "$CSV_DIR/tube_alternatives_to_label.csv" \
  --video_dir "$VIDEO_DIR" \
  --app_dir /app/web/tube_labeler
