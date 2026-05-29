#!/usr/bin/env python3
"""Create a minimal Raspberry Pi runtime bundle.

The bundle intentionally excludes lab artifacts, review packets, raw videos,
training outputs, and caches. It contains only the detector hot-path scripts,
Pi wrappers, service files, and deployment documentation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


BUNDLE_ROOT = "fpv-drone-vision-tracking-pi"
REQUIRED_FILES = [
    Path("scripts/tbd_motion_detector.py"),
    Path("scripts/motion_detector_v2.py"),
]
EXCLUDE_NAMES = {
    "__pycache__",
    ".DS_Store",
}
EXCLUDE_SUFFIXES = {
    ".pyc",
    ".pyo",
    ".tar",
    ".tgz",
    ".gz",
    ".zip",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo_root", default=str(Path(__file__).resolve().parents[1]))
    p.add_argument(
        "--out",
        required=True,
        help="Output .tar.gz path, or a directory path when --format dir is used.",
    )
    p.add_argument("--format", choices=("tar.gz", "dir"), default="tar.gz")
    p.add_argument("--force", action="store_true")
    return p.parse_args()


def should_exclude(path: Path) -> bool:
    if any(part in EXCLUDE_NAMES for part in path.parts):
        return True
    if path.suffix in EXCLUDE_SUFFIXES:
        return True
    return False


def iter_runtime_files(repo: Path) -> Iterable[Path]:
    for rel in REQUIRED_FILES:
        yield rel
    runtime_dir = repo / "raspberry_pi_runtime"
    for path in sorted(runtime_dir.rglob("*")):
        if path.is_dir():
            continue
        rel = path.relative_to(repo)
        if should_exclude(rel):
            continue
        yield rel


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(repo: Path) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for rel in iter_runtime_files(repo):
        if rel in seen:
            continue
        src = repo / rel
        if not src.exists():
            raise FileNotFoundError(f"required bundle file missing: {rel}")
        seen.add(rel)
        files.append(rel)
    return files


def stage_bundle(repo: Path, staging_root: Path, files: list[Path]) -> dict:
    bundle_dir = staging_root / BUNDLE_ROOT
    bundle_dir.mkdir(parents=True, exist_ok=True)
    manifest_files: list[dict] = []
    for rel in files:
        src = repo / rel
        dst = bundle_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        manifest_files.append(
            {
                "path": rel.as_posix(),
                "bytes": src.stat().st_size,
                "sha256": sha256_file(src),
            }
        )
    manifest = {
        "bundle_root": BUNDLE_ROOT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_count": len(manifest_files),
        "files": manifest_files,
        "excludes": [
            "artifacts/",
            "deploy_assets/",
            "results/",
            "raw videos",
            "review packets",
            "__pycache__/",
        ],
    }
    (bundle_dir / "bundle_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def create_tar(staging_root: Path, out_path: Path) -> None:
    with tarfile.open(out_path, "w:gz") as tar:
        tar.add(staging_root / BUNDLE_ROOT, arcname=BUNDLE_ROOT)


def create_dir(staging_root: Path, out_path: Path, force: bool) -> None:
    if out_path.exists():
        if not force:
            raise FileExistsError(f"{out_path} exists; pass --force to replace it")
        if out_path.is_dir():
            shutil.rmtree(out_path)
        else:
            out_path.unlink()
    shutil.copytree(staging_root / BUNDLE_ROOT, out_path)


def main() -> None:
    args = parse_args()
    repo = Path(args.repo_root).resolve()
    out = Path(args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not args.force:
        raise SystemExit(f"{out} exists; pass --force to replace it")
    files = collect_files(repo)
    with tempfile.TemporaryDirectory(prefix="fpv-pi-bundle-") as tmp:
        staging_root = Path(tmp)
        manifest = stage_bundle(repo, staging_root, files)
        if args.format == "tar.gz":
            if out.exists():
                out.unlink()
            create_tar(staging_root, out)
        else:
            create_dir(staging_root, out, args.force)
    print(json.dumps({"out": str(out), "file_count": manifest["file_count"]}, indent=2))


if __name__ == "__main__":
    main()
