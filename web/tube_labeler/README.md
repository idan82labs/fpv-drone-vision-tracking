# Drone Candidate Review Labeler

Local web UI for labeling `tube_alternatives_to_label.csv`.

Run:

```bash
.venv/bin/python scripts/tube_labeling_server.py \
  --host 0.0.0.0 \
  --port 8766 \
  --csv results/tube_alternative_review_packet_top16/tube_alternatives_to_label.csv \
  --video_dir /path/to/mp4_folder
```

The CSV is updated in place. The server creates one `.bak` backup next to the CSV before the first save.

The UI shows two overlays:

- Cyan box: the previously reviewed/reference drone position.
- Red box: the currently selected rank/candidate to label.

Choose `Drone target` only when the red candidate is on the cyan/reference drone.

Expected asset layout:

- The CSV has `overview_image` and `crop_sheet` paths relative to the repository root.
- Videos are named `<clip>.MP4` and stored in `--video_dir`.
- Labels are saved into `human_label` and `human_notes`.
