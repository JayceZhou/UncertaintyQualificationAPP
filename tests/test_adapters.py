import unittest
from pathlib import Path

import numpy as np

from inference import DFUNAdapter, EviFieldAdapter, evifield_arrays_to_frame, model_registry
from uqcore import analyze_niw_field


ROOT = Path(__file__).resolve().parents[1]


class AdapterTests(unittest.TestCase):
    def test_packaged_checkpoints_are_discovered(self):
        self.assertTrue(DFUNAdapter().available)
        self.assertTrue(EviFieldAdapter().available)
        self.assertEqual([item["name"] for item in model_registry()], ["DFUN", "EviField"])

    def test_evifield_expected_arrays_match_niw_analysis_contract(self):
        expected = np.load(
            ROOT / "system_handoff_evifield_era" / "expected_output" / "test_sample_evifield_era_outputs.npz"
        )
        frame = evifield_arrays_to_frame(
            expected["pred_field"], expected["kappa"], expected["nu"], expected["L"]
        )
        result = analyze_niw_field(frame).pixels
        covariance = expected["C_total"][0]
        self.assertEqual(len(result), 64 * 96)
        np.testing.assert_allclose(
            result["total_11"].to_numpy(), covariance[..., 0, 0].reshape(-1), rtol=2e-6, atol=0.05
        )
        np.testing.assert_allclose(
            result["total_12"].to_numpy(), covariance[..., 0, 1].reshape(-1), rtol=2e-6, atol=0.005
        )
        np.testing.assert_allclose(
            result["total_22"].to_numpy(), covariance[..., 1, 1].reshape(-1), rtol=2e-6, atol=0.02
        )


if __name__ == "__main__":
    unittest.main()
