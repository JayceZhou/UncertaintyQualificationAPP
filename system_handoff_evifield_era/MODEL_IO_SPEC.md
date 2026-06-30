# Model I/O Specification

## 1. Network Structure Code And Custom Layers

The full architecture is in:

- `code/model.py`
- `code/evidential.py`

Main classes/custom layers:

- `UniVecEDL`: full model wrapper.
- `SharedPyramidEncoder`: shared-weight RGB encoder.
- `OmniFusionTransformer`: cross-attention fusion module.
- `ConvGRUCell`: recurrent decoder cell.
- `DecoderStage`: multi-scale recurrent field decoder.
- `FinalNIWHead2D`: dense 2D NIW output head.
- `CovarianceRiskProjector`: optional CRP head, not used by this checkpoint.
- `unpack_niw_2d`: raw-to-structured NIW conversion.

This package uses:

```text
evidential_mode = niw
flow_support_aggregation = warp
niw_r = 1.0
niw_eps = 1e-6
niw_nu_min = 3.1
niw_nu_max = None
iters_per_scale = 4
use_risk_projector = False
```

## 2. Trained Parameter File

```text
checkpoints/evifield_era_optical_flow_latest_S_Mix.pt
```

This is the only parameter file required for the packaged optical-flow model.

## 3. Input Data Type And Channel Meaning

Task: optical flow between two RGB frames.

Input list:

```text
images[0] = frame 1 / reference image
images[1] = frame 2 / support image
```

Each frame:

```text
shape before batching: [3, H, W]
shape after batching:  [B, 3, H, W]
channel order: RGB
dtype: float32
value range before normalization: uint8/float 0..255 or float 0..1
value range after normalization: [-1, 1]
```

Normalization:

```python
image = image.float()
if image.max() > 1:
    image = image / 255.0
image = image * 2.0 - 1.0
```

The reference implementation is `code/preprocess.py`.

## 4. Image Or Field Data Size

Formal S+Mix training crop:

```text
416 x 960
```

The model supports arbitrary input height/width:

```text
input tensors: [B, 3, H, W]
output tensors: [B, ..., H, W]
```

Internally, the model pads the image to a multiple of 16 with replicate padding and crops the output back to the original input size.

If an external system resizes images before inference, the predicted flow is in resized image coordinates. To map flow back to the original image, resize the flow and multiply x/y components by the width/height scale factors. See `resize_flow_to_original()` in `code/preprocess.py`.

## 5. Resize, Crop, And Normalization

Recommended production inference:

```text
resize: none unless required by memory/latency
crop: none
random flip: none
normalization: RGB 0..255 or 0..1 -> [-1, 1]
```

Training used random crop/flip, but inference should be deterministic.

## 6. Input Tensor Shape

The model forward signature is:

```python
outputs = model(
    images=[frame1, frame2],
    task_type="optical_flow",
    iters_per_scale=4,
    use_niw=True,
)
```

Where:

```text
frame1: [B, 3, H, W]
frame2: [B, 3, H, W]
```

## 7. Output Tensor Channel Order

Primary outputs:

```text
outputs["pred_field"] : [B, 2, H, W]
  channel 0 = horizontal displacement u / dx / x
  channel 1 = vertical displacement v / dy / y

outputs["raw_niw"] : [B, 6, H, W]
  channel 0 = mu_x
  channel 1 = mu_y
  channel 2 = kappa_raw
  channel 3 = l11_raw
  channel 4 = l21_raw
  channel 5 = l22_raw
```

Structured NIW outputs:

```text
outputs["niw"]["mean"]    : [B, 2, H, W]
outputs["niw"]["kappa"]   : [B, 1, H, W]
outputs["niw"]["nu"]      : [B, 1, H, W]
outputs["niw"]["L"]       : [B, H, W, 2, 2]
outputs["niw"]["Psi"]     : [B, H, W, 2, 2]
outputs["niw"]["C_ale"]   : [B, H, W, 2, 2]
outputs["niw"]["C_epi"]   : [B, H, W, 2, 2]
outputs["niw"]["C_total"] : [B, H, W, 2, 2]
outputs["niw"]["U_total"] : [B, 1, H, W]
```

Legacy top-level covariance layout:

```text
outputs["total_cov"] : [B, 2, 2, H, W]
```

Use `outputs["niw"]["C_total"]` for new integrations because its last two axes are the covariance matrix.

## 8. Parameter Constraint Conversion

This checkpoint uses the 6-channel tied-nu NIW head.

Raw channels:

```python
mean = raw[:, 0:2]
kappa_raw = raw[:, 2:3]
l11_raw = raw[:, 3:4]
l21_raw = raw[:, 4:5]
l22_raw = raw[:, 5:6]
```

Constraint transforms:

```python
min_nu = 3.1
r = 1.0
eps = 1e-6
min_kappa = min_nu / r

kappa = min_kappa + softplus(kappa_raw) + eps
nu = kappa * r

l11 = softplus(l11_raw) + eps
l21 = l21_raw
l22 = softplus(l22_raw) + eps
```

Cholesky-like lower-triangular matrix:

```text
L = [[l11, 0],
     [l21, l22]]
```

For each pixel:

```python
base_psi = L @ L.T
Psi = nu * base_psi

rho = nu - d - 1 = nu - 3
C_ale = Psi / rho
C_epi = Psi / (kappa * rho)
C_total = C_ale + C_epi
U_total = 0.5 * trace(C_total)
```

`nu_min=3.1` keeps `rho=nu-3` positive.

## 9. Risk Readout

This handoff uses the final EviField-ERA analytic evidence score:

```python
d = 2
s_E = -log1p(kappa) - log1p(nu - d - 1 + eps)
```

Tensor shape:

```text
risk_score: [B, 1, H, W]
```

Interpretation:

```text
higher risk_score = higher predicted risk
```

Because the score is often negative, use ranking/order rather than absolute zero as a semantic threshold unless calibrated in the deployment domain.

## 10. Minimal Inference Example

```bash
python code/inference_minimal.py \
  --checkpoint checkpoints/evifield_era_optical_flow_latest_S_Mix.pt \
  --frame1 test_sample/frame1.png \
  --frame2 test_sample/frame2.png \
  --output expected_output/reproduced_outputs.npz \
  --device cuda
```

The `.npz` output contains:

```text
pred_field, raw_niw, kappa, nu, L, Psi, C_ale, C_epi, C_total, U_total, risk_score
```

## 11. Test Sample And Expected Output

Test input:

```text
test_sample/frame1.png
test_sample/frame2.png
```

Expected output:

```text
expected_output/test_sample_evifield_era_outputs.npz
expected_output/test_sample_evifield_era_outputs.summary.json
```

Expected summary:

```text
pred_field_shape = [1, 2, 64, 96]
raw_niw_shape    = [1, 6, 64, 96]
C_total_shape    = [1, 64, 96, 2, 2]
risk_score_shape = [1, 1, 64, 96]
pred_field_mean  = 1.3582022190
pred_field_std   = 0.8379982114
kappa_mean       = 3.6256673336
nu_mean          = 3.6256673336
U_total_mean     = 1390.0778808594
risk_score_mean  = -2.0010972023
```

