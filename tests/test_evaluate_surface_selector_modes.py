import unittest
from pathlib import Path

from scripts import evaluate_surface_selector_modes as modes


class EvaluateSurfaceSelectorModesTests(unittest.TestCase):
    def test_validate_score_value_rejects_blank_and_nonfinite_scores(self):
        path = Path("dummy/top_tubes.csv")

        self.assertEqual(modes.validate_score_value("0.42", "learned_score", path), "0.42")
        with self.assertRaises(SystemExit):
            modes.validate_score_value("", "learned_score", path)
        with self.assertRaises(SystemExit):
            modes.validate_score_value("nan", "learned_score", path)


if __name__ == "__main__":
    unittest.main()
