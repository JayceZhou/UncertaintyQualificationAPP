import unittest

import numpy as np
import pandas as pd

from uqcore.classification import analyze_mc_probabilities
from uqcore.niw import analyze_niw_field


class ClassificationTests(unittest.TestCase):
    def test_deterministic_predictions_have_zero_sampling_disagreement(self):
        rows = []
        for sample, probabilities, true_label in [
            ("a", [0.9, 0.1], "A"),
            ("b", [0.3, 0.7], "B"),
        ]:
            for pass_id in range(5):
                for label, probability in zip(["A", "B"], probabilities):
                    rows.append(
                        {
                            "sample_id": sample,
                            "pass_id": pass_id,
                            "class_label": label,
                            "probability": probability,
                            "true_label": true_label,
                        }
                    )
        result = analyze_mc_probabilities(pd.DataFrame(rows))
        np.testing.assert_allclose(result.samples["mutual_information"], 0.0, atol=1e-12)
        np.testing.assert_allclose(result.samples["variation_ratio"], 0.0)
        self.assertEqual(result.summary["accuracy"], 1.0)
        self.assertIn("risk", result.risk_coverage)

    def test_missing_columns_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "缺少必要列"):
            analyze_mc_probabilities(pd.DataFrame({"sample_id": [1]}))


class NIWTests(unittest.TestCase):
    def test_covariance_decomposition_matches_formula(self):
        data = pd.DataFrame(
            {
                "x": [0],
                "y": [0],
                "mean_1": [1.0],
                "mean_2": [2.0],
                "kappa": [2.0],
                "nu": [5.0],
                "l11": [2.0],
                "l21": [0.0],
                "l22": [1.0],
            }
        )
        row = analyze_niw_field(data, epsilon=0.0).pixels.iloc[0]
        self.assertAlmostEqual(row.aleatoric_11, 2.0)
        self.assertAlmostEqual(row.epistemic_11, 1.0)
        self.assertAlmostEqual(row.total_11, 3.0)
        self.assertAlmostEqual(row.total_22, 0.75)
        self.assertAlmostEqual(row.trace_uncertainty, 1.875)

    def test_invalid_degrees_of_freedom_are_rejected(self):
        data = pd.DataFrame(
            {
                "x": [0], "y": [0], "mean_1": [0], "mean_2": [0],
                "kappa": [1], "nu": [3], "l11": [1], "l21": [0], "l22": [1],
            }
        )
        with self.assertRaisesRegex(ValueError, "nu 必须大于 3"):
            analyze_niw_field(data)


if __name__ == "__main__":
    unittest.main()

