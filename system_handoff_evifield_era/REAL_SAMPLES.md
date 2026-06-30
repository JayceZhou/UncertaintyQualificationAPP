# Real Validation Samples

This package includes one real optical-flow validation sample and one real LTEM validation sample. These are not synthetic smoke tests.

## Optical Flow

Source:

```text
Dataset: Sintel training split, clean render pass
Scene: shaman_3
Sample ID: clean:shaman_3:000000
Frame 1 source: /data/zhousc/datasets/Sintel/training/clean/shaman_3/frame_0001.png
Frame 2 source: /data/zhousc/datasets/Sintel/training/clean/shaman_3/frame_0002.png
GT flow source: /data/zhousc/datasets/Sintel/training/flow/shaman_3/frame_0001.flo
```

Packaged files:

```text
real_samples/optical_flow/raw/frame_0001.png
real_samples/optical_flow/raw/frame_0002.png
real_samples/optical_flow/raw/frame_0001.flo
real_samples/optical_flow/preprocessed/model_input.pt
real_samples/optical_flow/preprocessed/target_valid.pt
real_samples/optical_flow/preprocessed/target_valid.npz
real_samples/optical_flow/preprocessed/sample_meta.json
real_samples/optical_flow/expected_output/real_sintel_shaman3_outputs.npz
real_samples/optical_flow/expected_output/real_sintel_shaman3_outputs.summary.json
```

Formal validation preprocessing:

```text
raw size: 436 x 1024
resize: 448 x 1024
center crop: top=16, left=32, height=416, width=960
normalization: RGB 0..255 or 0..1 -> [-1, 1]
```

Reproduce expected output:

```bash
python code/inference_minimal.py \
  --checkpoint checkpoints/evifield_era_optical_flow_latest_S_Mix.pt \
  --frame1 real_samples/optical_flow/raw/frame_0001.png \
  --frame2 real_samples/optical_flow/raw/frame_0002.png \
  --output real_samples/optical_flow/expected_output/reproduced_real_sintel_shaman3_outputs.npz \
  --device cuda \
  --resize-height 448 --resize-width 1024 \
  --center-crop-height 416 --center-crop-width 960
```

Expected summary:

```text
pred_field_shape = [1, 2, 416, 960]
raw_niw_shape    = [1, 6, 416, 960]
C_total_shape    = [1, 416, 960, 2, 2]
risk_score_shape = [1, 1, 416, 960]
sample_epe       = 0.4758573174
sample_mae       = 0.2968619466
valid_pixels     = 399360
output_sha256    = a25ff9f1c968dd3fc6b9db800605cb17bb3f6cd4026c54822c39cbfc25111c85
```

## LTEM

Source:

```text
Dataset: LTEMData validation split
Sample ID: 58125111111
Input order: U, I, O
U source: /data/zhousc/LTEM_data/LTEM_data/U/58125111111.png
I source: /data/zhousc/LTEM_data/LTEM_data/I/58125111111.png
O source: /data/zhousc/LTEM_data/LTEM_data/O/58125111111.png
Target color source: /data/zhousc/LTEM_data/LTEM_data/color/58125111111.jpg
```

Packaged files:

```text
real_samples/ltem/raw/U_58125111111.png
real_samples/ltem/raw/I_58125111111.png
real_samples/ltem/raw/O_58125111111.png
real_samples/ltem/raw/color_58125111111.jpg
real_samples/ltem/preprocessed/model_input.pt
real_samples/ltem/preprocessed/target_valid.pt
real_samples/ltem/preprocessed/target_valid.npz
real_samples/ltem/preprocessed/sample_meta.json
real_samples/ltem/expected_output/real_ltem_58125111111_outputs.npz
real_samples/ltem/expected_output/real_ltem_58125111111_outputs.summary.json
```

Formal validation preprocessing:

```text
raw size: 256 x 256
resize: 256 x 256
center crop: top=16, left=16, height=224, width=224
normalization: grayscale 0..255 or 0..1 -> 0..1
input order: U, I, O
```

Reproduce expected output:

```bash
python code/inference_ltem_minimal.py \
  --checkpoint checkpoints/evifield_era_ltem_latest_FT.pt \
  --under real_samples/ltem/raw/U_58125111111.png \
  --infocus real_samples/ltem/raw/I_58125111111.png \
  --over real_samples/ltem/raw/O_58125111111.png \
  --output real_samples/ltem/expected_output/reproduced_real_ltem_58125111111_outputs.npz \
  --device cuda
```

Expected summary:

```text
pred_field_shape = [1, 2, 224, 224]
raw_niw_shape    = [1, 6, 224, 224]
C_total_shape    = [1, 224, 224, 2, 2]
risk_score_shape = [1, 1, 224, 224]
sample_epe       = 0.5040577650
sample_mae       = 0.3226417005
valid_pixels     = 21285
output_sha256    = 459c9df094c97bf2cb4d6f27d3e6329552362c507b28872d0ca7f444ebfec7eb
```

Machine-readable summary:

```text
real_samples/real_samples_summary.json
real_samples/manifest.json
```

