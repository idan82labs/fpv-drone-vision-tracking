# Deployment

The labeling app can be deployed to Fly.io with the included Dockerfile.

## Local Smoke Test

```bash
python scripts/tube_labeling_server.py \
  --host 127.0.0.1 \
  --port 8768 \
  --csv deploy_assets/tube_hard_negative_review_packet_thr060_top8/tube_alternatives_to_label.csv \
  --video_dir deploy_assets/videos \
  --app_dir web/tube_labeler
```

## Fly.io

```bash
cp config/fly.example.toml fly.toml
fly apps create <app-name>
fly volumes create label_data --region fra --size 1
fly secrets set BASIC_AUTH_PASSWORD='<password>'
fly deploy
```

The example config sets `BASIC_AUTH_USERNAME=review`. Change it in `fly.toml` if needed.
The server refuses to bind to `0.0.0.0` without `BASIC_AUTH_PASSWORD` unless
`ALLOW_UNAUTHENTICATED=1` is explicitly set for a trusted private network.

The server reads:

- `BASIC_AUTH_USERNAME`
- `BASIC_AUTH_USER` as a legacy fallback
- `BASIC_AUTH_PASSWORD`
- `DATA_DIR`
- `PACKET_NAME`

Do not commit real passwords.

## Label Backup

Use the in-app `Export CSV` link or download:

```bash
curl -u review:'<password>' https://<app-name>.fly.dev/api/export -o tube_alternatives_labeled.csv
```

The Fly volume persists labels under `/data`, but exported CSVs should still be
pulled back into `artifacts/` or `deploy_assets/` before training.
