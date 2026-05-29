# Local Deploy Assets

`deploy_assets/` is local working data for the browser labeler and temporary
review deployments. It may contain raw/compressed videos, candidate crop sheets,
overviews, labeling CSVs, and backups.

The directory is ignored by Git because these files are large, change often,
and may contain source footage. For a shared labeling job, package the needed
CSV, crops, overviews, and videos separately or deploy them to the labeling
server storage volume.

Typical local labeler command:

```bash
python scripts/tube_labeling_server.py \
  --host 127.0.0.1 \
  --port 8768 \
  --csv deploy_assets/tube_hard_negative_review_packet_thr060_top8/tube_alternatives_to_label.csv \
  --video_dir deploy_assets/videos \
  --app_dir web/tube_labeler
```
