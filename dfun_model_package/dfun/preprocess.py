from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from scipy.signal import find_peaks, peak_widths
from scipy.stats import kurtosis, skew


NUM_PEAKS = 10
PHYSICAL_FEATURE_DIM = NUM_PEAKS * 4 + 5


def training_d_spacing_grid() -> np.ndarray:
    """Return the d-spacing grid used by the training/evaluation pipeline.

    The original code uses np.flip(np.linspace(0.889, 17.659, 5000)).
    Keep this descending direction for exact compatibility with the trained
    checkpoint.
    """
    return np.flip(np.linspace(0.889, 17.659, 5000)).astype(np.float32)


def normalize_intensity(intensity: np.ndarray, target_max: float = 100.0) -> np.ndarray:
    """Clip negative intensities and max-normalize to the 0-100 training scale."""
    y = np.asarray(intensity, dtype=np.float32).reshape(-1)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, 0.0, None)
    max_value = float(np.max(y)) if y.size else 0.0
    if max_value > 0:
        y = y / max_value * target_max
    return y.astype(np.float32)


def resample_to_training_grid(
    intensity: np.ndarray,
    source_grid: np.ndarray | None = None,
    target_grid: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Resample an input spectrum to the 5000-point training d-spacing grid.

    If source_grid is None and intensity already has length 5000, no resampling
    is applied. If source_grid is provided, linear interpolation is used.
    """
    target = training_d_spacing_grid() if target_grid is None else np.asarray(target_grid, dtype=np.float32)
    y = np.asarray(intensity, dtype=np.float32).reshape(-1)

    if source_grid is None:
        if y.shape[0] != target.shape[0]:
            raise ValueError(
                f"source_grid is required when intensity length is {y.shape[0]}, "
                f"expected {target.shape[0]}."
            )
        return y.astype(np.float32), target

    x = np.asarray(source_grid, dtype=np.float32).reshape(-1)
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"source_grid length {x.shape[0]} does not match intensity length {y.shape[0]}.")

    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    resampled = np.interp(target, x_sorted, y_sorted, left=0.0, right=0.0)
    return resampled.astype(np.float32), target


def extract_physical_features(d_spacing_grid: np.ndarray, intensity_array: np.ndarray, num_peaks: int = NUM_PEAKS) -> np.ndarray:
    """Extract the 45-dimensional peak/physical descriptor used by DFUN.

    Descriptor layout:
        For each of the top 10 peaks by prominence:
            [d_position, intensity, FWHM_in_d_units, asymmetry]
        Then 5 global statistics:
            [mean_intensity, std_intensity, skewness, kurtosis, intensity_centroid]
    """
    d_spacing_grid = np.asarray(d_spacing_grid, dtype=np.float32).reshape(-1)
    intensity_array = np.asarray(intensity_array, dtype=np.float32).reshape(-1)
    if d_spacing_grid.shape[0] != intensity_array.shape[0]:
        raise ValueError("d_spacing_grid and intensity_array must have the same length.")

    max_intensity = float(np.max(intensity_array)) if intensity_array.size else 0.0
    if max_intensity > 0:
        peaks, properties = find_peaks(intensity_array, prominence=max_intensity * 0.05)
    else:
        peaks, properties = np.array([], dtype=int), {"prominences": np.array([], dtype=np.float32)}

    peak_features = []
    if len(peaks) > 0:
        widths, _, left_ips, right_ips = peak_widths(intensity_array, peaks, rel_height=0.5)
        peak_intensities = properties["prominences"]
        sorted_indices = np.argsort(peak_intensities)[::-1]

        for i in range(min(num_peaks, len(sorted_indices))):
            idx = sorted_indices[i]
            peak_idx = peaks[idx]

            peak_top_d = d_spacing_grid[peak_idx]
            left_base_d = d_spacing_grid[int(left_ips[idx])]
            right_base_d = d_spacing_grid[int(right_ips[idx])]
            asymmetry = (peak_top_d - left_base_d) - (right_base_d - peak_top_d)

            d_val = d_spacing_grid[peak_idx]
            intensity_val = intensity_array[peak_idx]
            fwhm_val = widths[idx] * (d_spacing_grid[1] - d_spacing_grid[0])
            peak_features.extend([d_val, intensity_val, fwhm_val, asymmetry])

    padding_length = num_peaks * 4 - len(peak_features)
    if padding_length > 0:
        peak_features.extend([0.0] * padding_length)

    safe_intensity = np.maximum(intensity_array, 0.0)
    intensity_sum = float(np.sum(safe_intensity))
    centroid = float(np.sum(d_spacing_grid * safe_intensity) / intensity_sum) if intensity_sum > 0 else 0.0
    statistical_features = [
        float(np.mean(intensity_array)),
        float(np.std(intensity_array)),
        float(skew(intensity_array)),
        float(kurtosis(intensity_array)),
        centroid,
    ]

    features = np.asarray(peak_features + statistical_features, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    if features.shape[0] != PHYSICAL_FEATURE_DIM:
        raise RuntimeError(f"Expected physical feature dim {PHYSICAL_FEATURE_DIM}, got {features.shape[0]}.")
    return features


def prepare_inputs(intensity: np.ndarray, source_grid: np.ndarray | None = None) -> tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """Return model-ready tensors.

    Returns:
        raw_xrd: FloatTensor [1, 5000]
        physical_features: FloatTensor [1, 45]
        d_spacing_grid: numpy array [5000]
    """
    intensity_5000, d_grid = resample_to_training_grid(intensity, source_grid=source_grid)
    intensity_5000 = normalize_intensity(intensity_5000)
    physical = extract_physical_features(d_grid, intensity_5000)
    raw_xrd = torch.from_numpy(intensity_5000).float().unsqueeze(0)
    physical_features = torch.from_numpy(physical).float().unsqueeze(0)
    return raw_xrd, physical_features, d_grid


def load_sample_npz(path: str | Path, sample_index: int = 0) -> tuple[np.ndarray, np.ndarray | None, dict]:
    """Load one spectrum from a sample NPZ file.

    Supported keys:
        intensity: a single 1-D spectrum.
        d_spacing: optional d-spacing grid for intensity.
        features: an array [N, L] or [L]; sample_index selects one row.
        labels230: optional zero-based label index.
    """
    path = Path(path)
    data = np.load(path, allow_pickle=True)
    metadata = {"path": str(path), "sample_index": int(sample_index)}

    if "intensity" in data.files:
        intensity = np.asarray(data["intensity"], dtype=np.float32).reshape(-1)
    elif "features" in data.files:
        features = np.asarray(data["features"], dtype=np.float32)
        intensity = features[sample_index] if features.ndim == 2 else features.reshape(-1)
    else:
        raise KeyError(f"{path} must contain either 'intensity' or 'features'. Available keys: {data.files}")

    source_grid = np.asarray(data["d_spacing"], dtype=np.float32).reshape(-1) if "d_spacing" in data.files else None
    if "label_index" in data.files:
        metadata["label_index"] = int(np.asarray(data["label_index"]).reshape(-1)[0])
    elif "labels230" in data.files:
        labels = np.asarray(data["labels230"]).reshape(-1)
        metadata["label_index"] = int(labels[sample_index])
    if "space_group" in data.files:
        metadata["space_group"] = int(np.asarray(data["space_group"]).reshape(-1)[0])
    elif "label_index" in metadata:
        metadata["space_group"] = int(metadata["label_index"]) + 1

    return intensity, source_grid, metadata
