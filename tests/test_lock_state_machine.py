import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from scripts import evaluate_lock_state_machine as lock_sm


class LockStateMachineTests(unittest.TestCase):
    def test_load_candidates_uses_requested_score_and_rank_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            path.write_text(
                "\n".join(
                    [
                        "frame,rank,score,verified_score,x,y,w,h",
                        "10,1,0.10,2.0,1,1,3,3",
                        "10,2,0.20,5.0,2,2,3,3",
                        "10,9,0.99,99.0,9,9,3,3",
                        "11,1,0.30,1.5,4,4,3,3",
                    ]
                )
                + "\n"
            )

            out = lock_sm.load_candidates(path, "verified_score", max_rank=2)

        self.assertEqual(set(out), {10, 11})
        self.assertEqual(out[10].rank, 2)
        self.assertEqual(out[10].score, 5.0)
        self.assertEqual(out[10].track_score, 5.0)
        self.assertEqual(out[10].bbox, (2.0, 2.0, 3.0, 3.0))
        self.assertEqual(out[11].score, 1.5)

    def test_load_candidates_falls_back_to_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            path.write_text("frame,rank,score,x,y,w,h\n1,1,0.75,7,8,2,2\n")

            out = lock_sm.load_candidates(path, "missing_score_column", max_rank=80)

        self.assertEqual(out[1].score, 0.75)

    def test_load_candidates_keeps_separate_track_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            path.write_text("frame,rank,accept_score,verified_score,x,y,w,h\n1,1,0.25,14.0,7,8,2,2\n")

            out = lock_sm.load_candidates(path, "accept_score", max_rank=80, track_score_column="verified_score")

        self.assertEqual(out[1].score, 0.25)
        self.assertEqual(out[1].track_score, 14.0)

    def test_clip_filter_prevents_frame_collisions(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.csv"
            labels_path.write_text(
                "\n".join(
                    [
                        "clip,frame,visible,det_x,det_y,det_w,det_h",
                        "a,1,1,1,1,2,2",
                        "b,1,0,,,,",
                    ]
                )
                + "\n"
            )
            candidates_path = Path(tmp) / "candidates.csv"
            candidates_path.write_text(
                "\n".join(
                    [
                        "clip,frame,rank,score,x,y,w,h",
                        "a,1,1,0.5,1,1,2,2",
                        "b,1,1,0.9,9,9,2,2",
                    ]
                )
                + "\n"
            )

            labels = lock_sm.load_labels(labels_path, clip="a")
            candidates = lock_sm.load_candidates(candidates_path, "score", max_rank=80, clip="a")

        self.assertTrue(labels[1].visible)
        self.assertEqual(candidates[1].bbox, (1.0, 1.0, 2.0, 2.0))

    def test_state_machine_uses_acquire_score_then_track_score(self):
        labels = {
            1: lock_sm.Label(visible=True, bbox=(10.0, 10.0, 4.0, 4.0)),
            2: lock_sm.Label(visible=True, bbox=(11.0, 10.0, 4.0, 4.0)),
        }
        candidates = {
            1: lock_sm.Candidate(score=0.9, track_score=0.1, rank=1, bbox=(10.0, 10.0, 4.0, 4.0)),
            2: lock_sm.Candidate(score=0.1, track_score=9.0, rank=1, bbox=(11.0, 10.0, 4.0, 4.0)),
        }

        summary, rows = lock_sm.evaluate_rows(
            labels,
            candidates,
            acquire_threshold=0.8,
            track_threshold=8.0,
            acquire_hits=1,
            max_misses=0,
            max_jump_px=12.0,
            strict_tol_px=8.0,
            loose_tol_px=16.0,
            output_tentative=False,
            coast_output=False,
        )

        self.assertEqual(summary["visible_strict"], 2)
        self.assertEqual(rows[0]["reason"], "acquire_start")
        self.assertEqual(rows[1]["reason"], "track")
        self.assertEqual(rows[1]["candidate_score"], 0.1)
        self.assertEqual(rows[1]["track_score"], 9.0)

    def test_dual_score_threshold_grids_can_use_different_scales(self):
        with tempfile.TemporaryDirectory() as tmp:
            labels_path = Path(tmp) / "labels.csv"
            labels_path.write_text("frame,visible,det_x,det_y,det_w,det_h\n1,1,1,1,2,2\n2,1,1,1,2,2\n")
            candidates_path = Path(tmp) / "candidates.csv"
            candidates_path.write_text(
                "\n".join(
                    [
                        "frame,rank,score,verified_score,x,y,w,h",
                        "1,1,0.8,20,1,1,2,2",
                        "2,1,0.1,20,1,1,2,2",
                    ]
                )
                + "\n"
            )
            out_dir = Path(tmp) / "out"

            old_argv = __import__("sys").argv
            try:
                __import__("sys").argv = [
                    "evaluate_lock_state_machine.py",
                    "--labels",
                    str(labels_path),
                    "--candidates",
                    str(candidates_path),
                    "--out_dir",
                    str(out_dir),
                    "--score_column",
                    "score",
                    "--track_score_column",
                    "verified_score",
                    "--acquire_thresholds",
                    "0.7",
                    "--track_thresholds",
                    "12",
                    "--acquire_hits",
                    "1",
                    "--max_misses",
                    "0",
                    "--max_jump_px",
                    "8",
                ]
                with redirect_stdout(StringIO()):
                    lock_sm.main()
            finally:
                __import__("sys").argv = old_argv

            rows = (out_dir / "state_machine_sweep.csv").read_text().splitlines()

        self.assertEqual(len(rows), 2)
        self.assertIn("1.0", rows[1])


if __name__ == "__main__":
    unittest.main()
