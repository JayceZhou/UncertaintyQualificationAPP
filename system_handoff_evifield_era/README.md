# EviField-ERA System Handoff

This folder contains one deployable EviField-ERA optical-flow model package for system integration.

## Files

| Path | Purpose |
|---|---|
| `checkpoints/evifield_era_optical_flow_latest_S_Mix.pt` | Trained parameter file. |
| `code/model.py` | Network architecture and custom layers. |
| `code/evidential.py` | NIW parameter transforms, covariance decomposition, risk features. |
| `code/preprocess.py` | Input loading, normalization, optional resize, flow-resize helper. |
| `code/inference_minimal.py` | Minimal model loading and inference example. |
| `code/inference_ltem_minimal.py` | Minimal LTEM model loading and inference example. |
| `code/create_test_sample.py` | Creates the deterministic smoke-test image pair. |
| `test_sample/frame1.png`, `test_sample/frame2.png` | Test input pair. |
| `expected_output/test_sample_evifield_era_outputs.npz` | Expected tensors for the test pair. |
| `expected_output/test_sample_evifield_era_outputs.summary.json` | Expected output summary for quick validation. |
| `real_samples/` | One real Sintel optical-flow sample and one real LTEM sample with expected outputs. |
| `REAL_SAMPLES.md` | Source paths, preprocessing, and commands for real samples. |
| `MODEL_IO_SPEC.md` | Detailed input/output and NIW parameter specification. |
| `manifest.json` | Checkpoint hash, model settings, and test-output summary. |

## Model

- Model name for integration: `EviField-ERA optical-flow NIW`
- Checkpoint: `checkpoints/evifield_era_optical_flow_latest_S_Mix.pt`
- SHA256: `289d1a37cfc13225f6dc892986775e66db3c6ae520897641e6f9e71b5c615a5f`
- Task: dense 2D optical flow.
- Input: two RGB frames.
- Output: 2D vector field plus dense NIW evidence/covariance attributes.
- Risk readout: analytic ERA-refined evidence score `s^E = -log(1+kappa) - log(1+nu-d-1+eps)`, with `d=2`. Larger score means higher risk.

## Minimal Run

From this folder:

```bash
python code/inference_minimal.py \
  --checkpoint checkpoints/evifield_era_optical_flow_latest_S_Mix.pt \
  --frame1 test_sample/frame1.png \
  --frame2 test_sample/frame2.png \
  --output expected_output/reproduced_outputs.npz \
  --device cuda
```

If GPU is unavailable, use `--device cpu`; CPU inference is slower.

Expected smoke-test summary:

```text
pred_field_shape = [1, 2, 64, 96]
raw_niw_shape    = [1, 6, 64, 96]
C_total_shape    = [1, 64, 96, 2, 2]
risk_score_shape = [1, 1, 64, 96]
pred_field_mean  = 1.3582022190
kappa_mean       = 3.6256673336
nu_mean          = 3.6256673336
U_total_mean     = 1390.0778808594
risk_score_mean  = -2.0010972023
```

Small floating-point differences are expected across GPU/CPU and PyTorch/CUDA versions.

## Real Samples

Real validation samples are available under `real_samples/`:

- Optical flow: Sintel clean `shaman_3`, `frame_0001 -> frame_0002`.
- LTEM: LTEMData sample `58125111111`, input order `U/I/O`.

See `REAL_SAMPLES.md` for source paths, preprocessing details, reproduction commands, expected output hashes, and single-sample EPE/MAE.

## Notes For Integration

- The checkpoint is a PyTorch training checkpoint, not TorchScript/ONNX. Use `code/inference_minimal.py` as the reference loader.
- The checkpoint does not contain the unused `final_gaussian_head.*` weights. The loader allows only those missing keys and raises on any other mismatch.
- The model internally pads image sizes to multiples of 16 and crops outputs back to the input size.
- Formal S+Mix training used random crops of `416 x 960`, but inference supports arbitrary `H x W` subject to GPU memory.
- For production, keep preprocessing exactly as in `code/preprocess.py`: RGB image, float tensor, `0..255` or `0..1` converted to `[-1, 1]`.
