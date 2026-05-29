import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import joblib
import numpy as np

from scripts import apply_crop_stack_verifier as apply_crop


class DummyModel:
    def predict_proba(self, x):
        score = np.clip(np.mean(x, axis=1), 0.0, 1.0)
        return np.column_stack([1.0 - score, score])


class ApplyCropStackVerifierTests(unittest.TestCase):
    def test_predict_score_uses_positive_probability(self):
        scores = apply_crop.predict_score(DummyModel(), np.asarray([[0.2, 0.4], [1.0, 1.0]], dtype=np.float32))

        self.assertEqual(scores.shape, (2,))
        self.assertAlmostEqual(float(scores[0]), 0.3)
        self.assertAlmostEqual(float(scores[1]), 1.0)

    def test_load_bundle_rejects_missing_runtime_metadata(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.joblib"
            joblib.dump({"model": DummyModel()}, path)

            with self.assertRaises(SystemExit):
                apply_crop.load_bundle(path)


if __name__ == "__main__":
    unittest.main()
