import tempfile
import unittest
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
        self.assertEqual(out[10].bbox, (2.0, 2.0, 3.0, 3.0))
        self.assertEqual(out[11].score, 1.5)

    def test_load_candidates_falls_back_to_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "candidates.csv"
            path.write_text("frame,rank,score,x,y,w,h\n1,1,0.75,7,8,2,2\n")

            out = lock_sm.load_candidates(path, "missing_score_column", max_rank=80)

        self.assertEqual(out[1].score, 0.75)


if __name__ == "__main__":
    unittest.main()
