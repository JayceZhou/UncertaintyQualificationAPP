# DFUN Model Handoff Package

This folder contains the DFUN model package for system integration.

## Selected Checkpoint

Use:

```bash
checkpoints/dfun_gate_fls_exp_test4_model.pth
```

This is the final Gate + FLS model checkpoint:

- Model: CNN + Augmentation + Peaks (Gate) + FLS
- Original path: `training_results/exp_results/exp_test4_model.pth`
- Output classes: 230
- Label convention: zero-based model output index `0..229` maps to space group `1..230`
- Clean opXRD evaluation result from revision analysis:
  - Accuracy: 72.65%
  - Weighted F1: 71.79%
  - Macro F1: 51.13%
  - Top-3 Accuracy: 76.33%

## Folder Contents

```text
dfun_model_package/
  README.md
  requirements.txt
  checkpoints/
    dfun_gate_fls_exp_test4_model.pth
  dfun/
    dfun_model.py
    preprocess.py
    infer.py
    focal_loss_label_smoothing.py
  mappings/
    space_group_mapping.csv
    space_group_mapping.json
  samples/
    opxrd_public_sample.npz
    expected_output.json
  metadata/
    model_card.json
    sample_metadata.json
```

## Network Structure

Implementation:

```text
dfun/dfun_model.py
```

The model has two input channels:

1. CNN/raw diffraction branch
2. Peak/physical descriptor MLP branch

Fusion is gated:

```python
gated_xrd_features = g * xrd_features
gated_phys_features = (1.0 - g) * phys_features
```

So a larger `gate_weight` means stronger reliance on the CNN/raw-XRD branch.
A smaller `gate_weight` means stronger reliance on the peak/physical-feature branch.

There are no non-PyTorch custom layers in the inference model. The training loss
reference `FocalLossWithLabelSmoothing` is included in:

```text
dfun/focal_loss_label_smoothing.py
```

## Input Preprocessing

Implementation:

```text
dfun/preprocess.py
```

### Diffraction Curve Input

Supported sample format:

```text
.npz with:
  intensity: 1-D intensity array
  d_spacing: optional 1-D d-spacing grid
```

The packaged example is:

```text
samples/opxrd_public_sample.npz
```

### Input Length And Resampling

Training/evaluation data use a 5000-point d-spacing grid:

```python
np.flip(np.linspace(0.889, 17.659, 5000))
```

For system integration, resample the incoming spectrum to this 5000-point grid.
The model then internally interpolates raw XRD from `[B, 1, L]` to `[B, 1, 8500]`
before the CNN branch.

### Intensity Normalization

The training data are max-normalized to the range `[0, 100]`.
The packaged preprocessing does:

1. Convert NaN/Inf to 0
2. Clip negative intensity to 0
3. Divide by the maximum intensity
4. Multiply by 100

### Peak Feature Extraction

The physical branch receives 45 features:

- Top 10 peaks are detected with `scipy.signal.find_peaks`
- Peak threshold: `prominence >= max_intensity * 0.05`
- Top peaks are sorted by prominence
- For each peak:
  - d-position
  - intensity
  - FWHM in d-spacing units
  - asymmetry
- Global statistics:
  - mean intensity
  - standard deviation
  - skewness
  - kurtosis
  - intensity centroid

Total feature dimension:

```text
10 peaks * 4 features + 5 statistics = 45
```

## Tensor Shapes

External inputs to `Model.forward`:

```text
raw_xrd:            [batch_size, 5000] recommended
physical_features: [batch_size, 45]
```

Internal CNN input after interpolation:

```text
[batch_size, 1, 8500]
```

Output:

```text
logits:        [batch_size, 230]
probabilities: [batch_size, 230]
gate_weight:   [batch_size, 1]
```

## Class Mapping

Files:

```text
mappings/space_group_mapping.csv
mappings/space_group_mapping.json
```

Mapping rule:

```text
model output label_index = space_group - 1
space_group = label_index + 1
```

Example:

```text
label_index 147 -> space_group 148
```

## Dropout And MC Dropout

Dropout layers:

```text
cnn_branch.3       p = 0.4
cnn_branch.7       p = 0.4
cnn_branch.11      p = 0.4
mlp_branch.2       p = 0.3
classifier_head.2  p = 0.5
classifier_head.5  p = 0.5
```

MC Dropout inference:

1. Set the full model to eval mode.
2. Switch only `torch.nn.Dropout` modules to train mode.
3. Keep BatchNorm layers in eval mode.
4. Run multiple stochastic forward passes under `torch.no_grad()`.
5. Average probabilities across passes.

Packaged helper:

```python
from dfun_model import enable_mc_dropout
```

Command example:

```bash
python dfun/infer.py --mc-dropout --mc-passes 30
```

## Minimal Inference Example

Install dependencies in a Python environment with PyTorch, NumPy and SciPy:

```bash
pip install -r requirements.txt
```

Run deterministic inference:

```bash
python dfun/infer.py \
  --checkpoint checkpoints/dfun_gate_fls_exp_test4_model.pth \
  --sample samples/opxrd_public_sample.npz \
  --device cpu \
  --top-k 5
```

Run MC Dropout inference:

```bash
python dfun/infer.py \
  --checkpoint checkpoints/dfun_gate_fls_exp_test4_model.pth \
  --sample samples/opxrd_public_sample.npz \
  --device cpu \
  --top-k 5 \
  --mc-dropout \
  --mc-passes 30
```

## Public Test Sample And Expected Output

Sample file:

```text
samples/opxrd_public_sample.npz
```

Metadata:

```text
source_index: 334
is_augmented: false
true label_index: 147
true space_group: 148
intensity shape: [5000]
```

Expected deterministic output is saved in:

```text
samples/expected_output.json
```

Expected top-5:

```text
rank 1: label_index 147, space_group 148, probability 0.9959539175
rank 2: label_index 138, space_group 139, probability 0.0007730557
rank 3: label_index 12,  space_group 13,  probability 0.0002552992
rank 4: label_index 14,  space_group 15,  probability 0.0000638367
rank 5: label_index 2,   space_group 3,   probability 0.0000554363
```

Expected gate weight:

```text
0.1749756485
```

## Notes For Integration

- Use `model.eval()` for ordinary deterministic inference.
- Use softmax over logits to obtain probabilities.
- The probability vector has exactly 230 entries.
- Do not remap labels through crystal systems; the model predicts the 230
  individual space groups directly.
- Keep the preprocessing compatible with `dfun/preprocess.py`; changing peak
  detection, d-grid direction, or intensity scaling can change predictions.
