import argparse
import json
import tempfile
import unittest
from pathlib import Path

from raspberry_pi_runtime.make_pi_bundle import BUNDLE_ROOT, collect_files, stage_bundle
from raspberry_pi_runtime.production_gate import evaluate


class PiBundleTest(unittest.TestCase):
    def test_bundle_manifest_includes_runtime_files_and_excludes_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo = Path(tmp) / "repo"
            (repo / "scripts").mkdir(parents=True)
            (repo / "raspberry_pi_runtime" / "__pycache__").mkdir(parents=True)
            (repo / "scripts" / "tbd_motion_detector.py").write_text("print('detector')\n")
            (repo / "scripts" / "motion_detector_v2.py").write_text("print('base')\n")
            (repo / "raspberry_pi_runtime" / "run_pi_detector.py").write_text("print('run')\n")
            (repo / "raspberry_pi_runtime" / "ignored.pyc").write_bytes(b"cache")
            (repo / "raspberry_pi_runtime" / "__pycache__" / "ignored.pyc").write_bytes(b"cache")

            files = collect_files(repo)
            rels = {path.as_posix() for path in files}

            self.assertIn("scripts/tbd_motion_detector.py", rels)
            self.assertIn("scripts/motion_detector_v2.py", rels)
            self.assertIn("raspberry_pi_runtime/run_pi_detector.py", rels)
            self.assertNotIn("raspberry_pi_runtime/ignored.pyc", rels)

            staging = Path(tmp) / "stage"
            manifest = stage_bundle(repo, staging, files)

            self.assertEqual(manifest["file_count"], 3)
            self.assertTrue((staging / BUNDLE_ROOT / "bundle_manifest.json").exists())


class ProductionGateTest(unittest.TestCase):
    def _args(self, run_dir: Path, **overrides):
        values = {
            "run_dir": str(run_dir),
            "report": "",
            "telemetry_jsonl": "",
            "out": "",
            "budget_ms": 33.3,
            "max_wall_ms": 100.0,
            "min_frames": 3,
            "max_report_frames": 0,
            "require_stream_only": True,
            "require_telemetry": True,
            "allow_fail": False,
        }
        values.update(overrides)
        return argparse.Namespace(**values)

    def test_gate_passes_bounded_streaming_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "report.json").write_text(
                json.dumps(
                    {
                        "report_mode": "summary",
                        "stream_only": True,
                        "summary": {
                            "n_processed": 3,
                            "report_frames_stored": 0,
                            "p95_wall_ms_per_frame": 20.0,
                            "p99_wall_ms_per_frame": 21.0,
                            "max_wall_ms_per_frame": 22.0,
                            "selected_output_frame_rate": 0.5,
                        },
                    }
                )
            )
            (run_dir / "telemetry.jsonl").write_text(
                "\n".join(
                    [
                        json.dumps({"wall_ms": 19, "process_ms": 18, "status": "selected", "selected": True}),
                        json.dumps({"wall_ms": 20, "process_ms": 19, "status": "no_target", "selected": False}),
                        json.dumps({"wall_ms": 21, "process_ms": 20, "status": "selected", "selected": True}),
                    ]
                )
                + "\n"
            )

            result = evaluate(self._args(run_dir))

            self.assertTrue(result["passed"])
            self.assertEqual(result["telemetry_summary"]["records"], 3)

    def test_gate_fails_missing_telemetry_and_latency_tail(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            (run_dir / "report.json").write_text(
                json.dumps(
                    {
                        "report_mode": "summary",
                        "stream_only": True,
                        "summary": {
                            "n_processed": 3,
                            "report_frames_stored": 0,
                            "p95_wall_ms_per_frame": 20.0,
                            "p99_wall_ms_per_frame": 40.0,
                            "max_wall_ms_per_frame": 120.0,
                        },
                    }
                )
            )

            result = evaluate(self._args(run_dir))
            failed = {check["name"] for check in result["checks"] if not check["passed"]}

            self.assertFalse(result["passed"])
            self.assertIn("p99_wall_budget", failed)
            self.assertIn("max_wall_budget", failed)
            self.assertIn("telemetry_present", failed)


if __name__ == "__main__":
    unittest.main()
