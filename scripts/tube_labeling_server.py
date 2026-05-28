#!/usr/bin/env python3
"""Local web labeler for top-tube alternative review packets."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import hmac
import json
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
from collections import OrderedDict
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "results/tube_alternative_review_packet_top16/tube_alternatives_to_label.csv"
DEFAULT_APP_DIR = REPO_ROOT / "web/tube_labeler"
DEFAULT_VIDEO_DIR = Path("/Users/idant/Downloads")
LABEL_FIELDS = [
    "human_label",
    "human_notes",
    "frame_target_bbox",
    "frame_target_visible",
    "frame_target_notes",
    "frame_target_coord_space",
]


LABEL_DEFS = [
    {
        "id": "target",
        "name": "Drone target",
        "hint": "The marked candidate is on the drone.",
        "key": "1",
    },
    {
        "id": "near_target_wrong_center",
        "name": "Close but wrong",
        "hint": "Near the drone, but the box is not centered enough.",
        "key": "2",
    },
    {
        "id": "static_hotspot",
        "name": "Fixed dot/speck",
        "hint": "A fixed background dot, sensor mark, or repeated speck.",
        "key": "3",
    },
    {
        "id": "line_attached",
        "name": "Pole/tree/edge",
        "hint": "Attached to a pole, tree, branch, wire, road edge, or line.",
        "key": "4",
    },
    {
        "id": "parallax_edge",
        "name": "Motion-warp edge",
        "hint": "A terrain, tree, cloud, or object edge caused by camera motion.",
        "key": "5",
    },
    {
        "id": "boundary_artifact",
        "name": "Skyline/boundary",
        "hint": "A skyline, horizon, roofline, or sky/ground boundary.",
        "key": "6",
    },
    {
        "id": "appearance_blob",
        "name": "Blob, not drone",
        "hint": "Looks like a small object, but is not the drone.",
        "key": "7",
    },
    {
        "id": "terrain_texture",
        "name": "Terrain/texture",
        "hint": "Grass, field rows, road texture, trees, buildings, or clutter.",
        "key": "8",
    },
    {
        "id": "noise",
        "name": "Video noise",
        "hint": "Compression, blur, glare, interpolation, or isolated video noise.",
        "key": "9",
    },
    {
        "id": "uncertain",
        "name": "Unsure",
        "hint": "Cannot decide from this frame and video context.",
        "key": "0",
    },
]


def _safe_int(value: str | int | None, default: int = 0) -> int:
    try:
        return int(float(value or default))
    except (TypeError, ValueError):
        return default


def _safe_float(value: str | float | None, default: float = 0.0) -> float:
    try:
        return float(value if value not in (None, "") else default)
    except (TypeError, ValueError):
        return default


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _video_metadata(path: Path) -> dict:
    meta = {
        "exists": path.exists(),
        "fps": 30.0,
        "frame_count": None,
        "duration_seconds": None,
        "width": None,
        "height": None,
    }
    if not path.exists():
        return meta
    try:
        import cv2  # type: ignore

        cap = cv2.VideoCapture(str(path))
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            meta.update(
                {
                    "fps": float(fps) if fps > 0 else 30.0,
                    "frame_count": frame_count or None,
                    "duration_seconds": (frame_count / fps) if fps > 0 and frame_count else None,
                    "width": width or None,
                    "height": height or None,
                }
            )
        cap.release()
    except Exception:
        pass
    return meta


def _bbox_from_text(text: str | None) -> tuple[int, int, int, int] | None:
    if not text:
        return None
    try:
        vals = ast.literal_eval(text)
    except Exception:
        return None
    if not isinstance(vals, (list, tuple)) or len(vals) != 4:
        return None
    try:
        x, y, w, h = [int(round(float(v))) for v in vals]
    except (TypeError, ValueError):
        return None
    return x, y, max(1, w), max(1, h)


def _read_frame(video_path: Path, frame_no: int, downscale: float = 0.5):
    import cv2  # type: ignore

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        return None
    if downscale != 1.0:
        frame = cv2.resize(frame, None, fx=downscale, fy=downscale, interpolation=cv2.INTER_AREA)
    return frame


def _encode_jpeg(img) -> bytes:
    import cv2  # type: ignore

    ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    if not ok:
        raise ValueError("failed to encode jpeg")
    return encoded.tobytes()


class TubeLabelStore:
    def __init__(self, csv_path: Path, video_dir: Path):
        self.csv_path = csv_path.resolve()
        self.video_dir = video_dir.resolve()
        self.lock = threading.Lock()
        self.rows: list[dict] = []
        self.fieldnames: list[str] = []
        self.video_meta: dict[str, dict] = {}
        self.backup_path = self.csv_path.with_name(self.csv_path.name + ".bak")
        self.load()

    def load(self) -> None:
        with self.lock:
            with self.csv_path.open(newline="") as f:
                reader = csv.DictReader(f)
                self.fieldnames = list(reader.fieldnames or [])
                for field in LABEL_FIELDS:
                    if field not in self.fieldnames:
                        self.fieldnames.append(field)
                self.rows = []
                for row in reader:
                    for field in LABEL_FIELDS:
                        row.setdefault(field, "")
                    self.rows.append(row)
            self._refresh_video_meta()

    def _refresh_video_meta(self) -> None:
        clips = sorted({row.get("clip", "") for row in self.rows if row.get("clip")})
        self.video_meta = {}
        for clip in clips:
            self.video_meta[clip] = _video_metadata(self.video_dir / f"{clip}.MP4")

    def _ensure_backup(self) -> None:
        if not self.backup_path.exists():
            shutil.copy2(self.csv_path, self.backup_path)

    def _save_locked(self) -> None:
        self._ensure_backup()
        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=self.csv_path.name + ".",
            suffix=".tmp",
            dir=str(self.csv_path.parent),
            text=True,
        )
        try:
            with os.fdopen(tmp_fd, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.rows)
            os.replace(tmp_name, self.csv_path)
        finally:
            if os.path.exists(tmp_name):
                os.unlink(tmp_name)

    def update_row(self, row_id: int, human_label: str | None, human_notes: str | None) -> dict:
        with self.lock:
            if row_id < 0 or row_id >= len(self.rows):
                raise IndexError("row id out of range")
            if human_label is not None:
                valid = {label["id"] for label in LABEL_DEFS}
                if human_label and human_label not in valid:
                    raise ValueError(f"unknown label: {human_label}")
                self.rows[row_id]["human_label"] = human_label
            if human_notes is not None:
                self.rows[row_id]["human_notes"] = human_notes
            self._save_locked()
            return self.serialized_row(row_id)

    def clear_checkpoint(self, clip: str, frame: str) -> dict:
        with self.lock:
            changed = 0
            for row in self.rows:
                if row.get("clip") == clip and str(row.get("frame")) == str(frame):
                    if row.get("human_label") or row.get("human_notes"):
                        changed += 1
                    row["human_label"] = ""
                    row["human_notes"] = ""
            if changed:
                self._save_locked()
            return self.state()

    def update_frame_target(
        self,
        clip: str,
        frame: str,
        frame_target_bbox: str | None,
        frame_target_visible: str | None,
        frame_target_notes: str | None,
    ) -> dict:
        visible = (frame_target_visible or "").strip()
        if visible not in {"", "yes", "no", "uncertain"}:
            raise ValueError(f"unknown frame target visibility: {visible}")

        bbox_text = (frame_target_bbox or "").strip()
        if bbox_text and _bbox_from_text(bbox_text) is None:
            raise ValueError("bad frame target bbox")

        with self.lock:
            changed = 0
            for row in self.rows:
                if row.get("clip") == clip and str(row.get("frame")) == str(frame):
                    row["frame_target_bbox"] = bbox_text
                    row["frame_target_visible"] = visible
                    row["frame_target_notes"] = frame_target_notes or ""
                    row["frame_target_coord_space"] = "overview_image"
                    changed += 1
            if not changed:
                raise ValueError("checkpoint not found")
            self._save_locked()
            return self.state()

    def video_path(self, clip: str) -> Path:
        path = (self.video_dir / f"{clip}.MP4").resolve()
        if not _is_relative_to(path, self.video_dir):
            raise ValueError("video path outside video directory")
        return path

    def image_path(self, raw_path: str) -> Path:
        decoded = unquote(raw_path)
        path = Path(decoded)
        if not path.is_absolute():
            path = REPO_ROOT / path
        path = path.resolve()
        data_dir = Path(os.environ.get("DATA_DIR", "/data")).resolve()
        if not _is_relative_to(path, REPO_ROOT) and not _is_relative_to(path, data_dir):
            raise ValueError("image path outside repository")
        return path

    def serialized_row(self, row_id: int) -> dict:
        row = dict(self.rows[row_id])
        frame = _safe_int(row.get("frame"))
        rank = _safe_int(row.get("rank"))
        clip = row.get("clip", "")
        video = self.video_meta.get(clip, {})
        fps = float(video.get("fps") or 30.0)
        row.update(
            {
                "id": row_id,
                "frame_int": frame,
                "rank_int": rank,
                "verified_score_float": _safe_float(row.get("verified_score")),
                "raw_score_float": _safe_float(row.get("raw_score")),
                "tube_verifier_score_float": _safe_float(row.get("tube_verifier_score")),
                "time_seconds": frame / fps if fps > 0 else 0,
                "overview_url": "/media/image?path=" + quote(row.get("overview_image", "")),
                "crop_url": "/media/image?path=" + quote(row.get("crop_sheet", "")),
                "selected_overview_url": f"/media/selected_overview?id={row_id}",
                "candidate_crop_url": f"/media/candidate_crop?id={row_id}",
            }
        )
        return row

    def selected_overview_jpeg(self, row_id: int) -> bytes:
        import cv2  # type: ignore

        with self.lock:
            if row_id < 0 or row_id >= len(self.rows):
                raise IndexError("row id out of range")
            row = dict(self.rows[row_id])

        path = self.image_path(row.get("overview_image", ""))
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError("failed to read overview image")
        bbox = _bbox_from_text(row.get("bbox"))
        if bbox is not None:
            x, y, w, h = bbox
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(img.shape[1] - 1, x + w), min(img.shape[0] - 1, y + h)
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 0, 255), 3)
            cv2.rectangle(img, (x0, y0), (x1, y1), (255, 255, 255), 1)
            label = f"selected r{row.get('rank', '?')}"
            cv2.putText(
                img,
                label,
                (x0, max(18, y0 - 7)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.44,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
        return _encode_jpeg(img)

    def candidate_crop_jpeg(self, row_id: int, pad: int = 28, out_size: int = 360) -> bytes:
        import cv2  # type: ignore
        import numpy as np  # type: ignore

        with self.lock:
            if row_id < 0 or row_id >= len(self.rows):
                raise IndexError("row id out of range")
            row = dict(self.rows[row_id])

        bbox = _bbox_from_text(row.get("bbox"))
        if bbox is None:
            raise ValueError("bad bbox")
        frame = _read_frame(self.video_path(row.get("clip", "")), _safe_int(row.get("frame")), downscale=0.5)
        if frame is None:
            raise ValueError("failed to read video frame")

        h_img, w_img = frame.shape[:2]
        x, y, w, h = bbox
        x0 = max(0, x - pad)
        y0 = max(0, y - pad)
        x1 = min(w_img, x + w + pad)
        y1 = min(h_img, y + h + pad)
        crop = frame[y0:y1, x0:x1].copy()
        if crop.size == 0:
            crop = np.zeros((64, 64, 3), dtype=np.uint8)
            x, y, w, h = 24, 24, 16, 16
            x0, y0 = 0, 0

        sx, sy = x - x0, y - y0
        cv2.rectangle(crop, (sx, sy), (sx + w, sy + h), (0, 0, 255), 2)
        cv2.rectangle(crop, (sx, sy), (sx + w, sy + h), (255, 255, 255), 1)

        tile = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_NEAREST)
        cv2.rectangle(tile, (0, 0), (out_size, 32), (0, 0, 0), -1)
        label = f"Rank {row.get('rank', '?')} selected candidate"
        cv2.putText(tile, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        return _encode_jpeg(tile)

    def progress(self) -> dict:
        total = len(self.rows)
        labeled = sum(1 for row in self.rows if row.get("human_label"))
        target = sum(1 for row in self.rows if row.get("human_label") == "target")
        uncertain = sum(1 for row in self.rows if row.get("human_label") == "uncertain")
        return {
            "total_rows": total,
            "labeled_rows": labeled,
            "unlabeled_rows": total - labeled,
            "target_rows": target,
            "uncertain_rows": uncertain,
            "percent": round((labeled / total) * 100, 1) if total else 0,
        }

    def state(self) -> dict:
        checkpoints: OrderedDict[tuple[str, str], dict] = OrderedDict()
        rows = [self.serialized_row(i) for i in range(len(self.rows))]
        for row in rows:
            key = (row.get("clip", ""), str(row.get("frame", "")))
            checkpoint = checkpoints.get(key)
            clip = key[0]
            if checkpoint is None:
                frame = _safe_int(row.get("frame"))
                video = self.video_meta.get(clip, {})
                fps = float(video.get("fps") or 30.0)
                checkpoint = {
                    "key": f"{clip}:{frame}",
                    "clip": clip,
                    "clip_short": clip[:8],
                    "frame": frame,
                    "checkpoint_label": row.get("checkpoint_label", ""),
                    "notes": row.get("notes", ""),
                    "overview_url": row.get("overview_url", ""),
                    "crop_url": row.get("crop_url", ""),
                    "video_url": "/media/video?clip=" + quote(clip),
                    "video_exists": bool(video.get("exists")),
                    "fps": fps,
                    "time_seconds": frame / fps if fps > 0 else 0,
                    "duration_seconds": video.get("duration_seconds"),
                    "video_width": video.get("width"),
                    "video_height": video.get("height"),
                    "frame_target_bbox": row.get("frame_target_bbox", ""),
                    "frame_target_visible": row.get("frame_target_visible", ""),
                    "frame_target_notes": row.get("frame_target_notes", ""),
                    "frame_target_coord_space": row.get("frame_target_coord_space", "overview_image"),
                    "rows": [],
                    "row_ids": [],
                    "labeled_count": 0,
                    "target_count": 0,
                }
                checkpoints[key] = checkpoint
            checkpoint["rows"].append(row)
            checkpoint["row_ids"].append(row["id"])
            if row.get("human_label"):
                checkpoint["labeled_count"] += 1
            if row.get("human_label") == "target":
                checkpoint["target_count"] += 1

        checkpoint_list = list(checkpoints.values())
        for checkpoint in checkpoint_list:
            checkpoint["row_count"] = len(checkpoint["rows"])
            checkpoint["complete"] = checkpoint["labeled_count"] == checkpoint["row_count"]
            checkpoint["has_target"] = checkpoint["target_count"] > 0
        return {
            "csv_path": str(self.csv_path),
            "backup_path": str(self.backup_path),
            "video_dir": str(self.video_dir),
            "labels": LABEL_DEFS,
            "progress": self.progress(),
            "checkpoints": checkpoint_list,
            "generated_at": time.time(),
        }


class LabelingHandler(BaseHTTPRequestHandler):
    server_version = "TubeLabelingServer/1.0"

    @property
    def store(self) -> TubeLabelStore:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def app_dir(self) -> Path:
        return self.server.app_dir  # type: ignore[attr-defined]

    def log_message(self, fmt: str, *args) -> None:
        print(f"{self.address_string()} - {fmt % args}")

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path != "/healthz" and not self._authorized():
                return
            if parsed.path in ("", "/"):
                self._send_static(self.app_dir / "index.html")
            elif parsed.path.startswith("/static/"):
                name = parsed.path.removeprefix("/static/")
                self._send_static((self.app_dir / name).resolve())
            elif parsed.path == "/api/state":
                self._send_json(self.store.state())
            elif parsed.path == "/api/export":
                self._send_csv_export()
            elif parsed.path == "/healthz":
                self._send_json({"ok": True})
            elif parsed.path == "/media/image":
                qs = parse_qs(parsed.query)
                raw_path = qs.get("path", [""])[0]
                self._send_static(self.store.image_path(raw_path))
            elif parsed.path == "/media/selected_overview":
                qs = parse_qs(parsed.query)
                row_id = int(qs.get("id", ["-1"])[0])
                self._send_bytes(self.store.selected_overview_jpeg(row_id), "image/jpeg")
            elif parsed.path == "/media/candidate_crop":
                qs = parse_qs(parsed.query)
                row_id = int(qs.get("id", ["-1"])[0])
                self._send_bytes(self.store.candidate_crop_jpeg(row_id), "image/jpeg")
            elif parsed.path == "/media/video":
                qs = parse_qs(parsed.query)
                clip = qs.get("clip", [""])[0]
                self._send_video(self.store.video_path(clip))
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error_json(exc)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if not self._authorized():
                return
            payload = self._read_json()
            if parsed.path == "/api/label":
                row = self.store.update_row(
                    int(payload.get("id")),
                    payload.get("human_label") if "human_label" in payload else None,
                    payload.get("human_notes") if "human_notes" in payload else None,
                )
                self._send_json({"ok": True, "row": row, "progress": self.store.progress()})
            elif parsed.path == "/api/clear_checkpoint":
                state = self.store.clear_checkpoint(str(payload.get("clip", "")), str(payload.get("frame", "")))
                self._send_json({"ok": True, "state": state})
            elif parsed.path == "/api/frame_target":
                state = self.store.update_frame_target(
                    str(payload.get("clip", "")),
                    str(payload.get("frame", "")),
                    payload.get("frame_target_bbox") if "frame_target_bbox" in payload else None,
                    payload.get("frame_target_visible") if "frame_target_visible" in payload else None,
                    payload.get("frame_target_notes") if "frame_target_notes" in payload else None,
                )
                self._send_json({"ok": True, "state": state})
            else:
                self.send_error(HTTPStatus.NOT_FOUND)
        except Exception as exc:
            self._send_error_json(exc)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        return json.loads(body.decode("utf-8") or "{}")

    def _authorized(self) -> bool:
        username = os.environ.get("BASIC_AUTH_USER", "")
        password = os.environ.get("BASIC_AUTH_PASSWORD", "")
        if not username and not password:
            return True
        header = self.headers.get("Authorization", "")
        expected_raw = f"{username}:{password}".encode("utf-8")
        expected = "Basic " + base64.b64encode(expected_raw).decode("ascii")
        if hmac.compare_digest(header, expected):
            return True
        self.send_response(HTTPStatus.UNAUTHORIZED)
        self.send_header("WWW-Authenticate", 'Basic realm="Drone Candidate Review"')
        self.send_header("Content-Length", "0")
        self.end_headers()
        return False

    def _send_json(self, data: dict, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_error_json(self, exc: Exception) -> None:
        self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)

    def _send_static(self, path: Path) -> None:
        path = path.resolve()
        if not (_is_relative_to(path, self.app_dir) or _is_relative_to(path, REPO_ROOT)):
            raise ValueError("static path outside allowed directories")
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        content_type = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_csv_export(self) -> None:
        data = self.store.csv_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/csv; charset=utf-8")
        self.send_header("Content-Disposition", 'attachment; filename="tube_alternatives_labeled.csv"')
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_bytes(self, data: bytes, content_type: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_video(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        file_size = path.stat().st_size
        range_header = self.headers.get("Range", "")
        start = 0
        end = file_size - 1
        status = HTTPStatus.OK

        match = re.match(r"bytes=(\d*)-(\d*)", range_header)
        if match:
            if match.group(1):
                start = int(match.group(1))
            if match.group(2):
                end = int(match.group(2))
            end = min(end, file_size - 1)
            status = HTTPStatus.PARTIAL_CONTENT

        if start >= file_size or end < start:
            self.send_response(HTTPStatus.REQUESTED_RANGE_NOT_SATISFIABLE)
            self.send_header("Content-Range", f"bytes */{file_size}")
            self.end_headers()
            return

        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == HTTPStatus.PARTIAL_CONTENT:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        with path.open("rb") as f:
            f.seek(start)
            remaining = length
            while remaining > 0:
                chunk = f.read(min(1024 * 1024, remaining))
                if not chunk:
                    break
                try:
                    self.wfile.write(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    return
                remaining -= len(chunk)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--video_dir", type=Path, default=DEFAULT_VIDEO_DIR)
    parser.add_argument("--app_dir", type=Path, default=DEFAULT_APP_DIR)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    store = TubeLabelStore(args.csv, args.video_dir)
    server = ThreadingHTTPServer((args.host, args.port), LabelingHandler)
    server.store = store  # type: ignore[attr-defined]
    server.app_dir = args.app_dir.resolve()  # type: ignore[attr-defined]

    url = f"http://{args.host}:{args.port}"
    print(f"Tube labeler: {url}")
    print(f"CSV: {store.csv_path}")
    print(f"Backup: {store.backup_path}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping labeler.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
