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


class DummyMultiClassModel:
    classes_ = np.asarray(["T", "S", "E", "H", "G"])

    def predict_proba(self, x):
        return np.asarray(
            [
                [0.70, 0.10, 0.10, 0.05, 0.05],
                [0.05, 0.60, 0.20, 0.05, 0.10],
            ],
            dtype=np.float32,
        )[: x.shape[0]]


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

    def test_class_probabilities_maps_named_outputs(self):
        x = np.ones((2, 3), dtype=np.float32)
        probs = apply_crop.class_probabilities(DummyMultiClassModel(), x, ["T", "S", "E", "H", "G"])

        self.assertAlmostEqual(float(probs["T"][0]), 0.70, places=5)
        self.assertAlmostEqual(float(probs["S"][1]), 0.60, places=5)
        self.assertAlmostEqual(float(probs["G"][1]), 0.10, places=5)
        self.assertLess(apply_crop.bounded_logit(0.1), 0.0)
        self.assertGreater(apply_crop.bounded_logit(0.9), 0.0)


if __name__ == "__main__":
    unittest.main()
