#!/usr/bin/env python3
"""Minimal EviField-ERA optical-flow inference example.

This script loads one checkpoint, runs two input frames, and exports the mean
flow, NIW evidence/covariance tensors, and analytic ERA risk score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from model import UniVecEDL
from preprocess import load_rgb_image, prepare_optical_flow_pair


EPS = 1e-6


def load_evifield_era(checkpoint_path: str | Path, device: str | torch.device = "cuda") -> tuple[UniVecEDL, dict[str, Any]]:
    checkpoint_path = Path(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    runtime = checkpoint.get("runtime_config_effective") or checkpoint.get("config", {}).get("runtime", {})
    model = UniVecEDL(
        evidential_mode=runtime.get("evidential_mode", "niw"),
        flow_support_aggregation=runtime.get("flow_support_aggregation", "warp"),
        niw_r=float(runtime.get("niw_r", 1.0)),
        niw_eps=float(runtime.get("niw_eps", EPS)),
        niw_nu_max=runtime.get("niw_nu_max", None),
        niw_nu_min=float(runtime.get("niw_nu_min", 3.1)),
        use_risk_projector=bool(runtime.get("use_risk_projector", False)),
        risk_projector_features=runtime.get("risk_projector_features", "covariance"),
    )
    state = checkpoint["model"]
    if any(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    load_result = model.load_state_dict(state, strict=False)
    unexpected = list(load_result.unexpected_keys)
    missing = [key for key in load_result.missing_keys if not key.startswith("final_gaussian_head.")]
    if missing or unexpected:
        raise RuntimeError(f"State-dict mismatch. missing={missing}, unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model, checkpoint


def analytic_evidence_score(kappa: torch.Tensor, nu: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """ERA-refined analytic evidence risk score s^E. Higher means higher risk."""
    safe_kappa = torch.nan_to_num(kappa.float(), nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    safe_nu = torch.nan_to_num(nu.float(), nan=3.0 + eps, posinf=3.0 + eps, neginf=3.0 + eps)
    nu_minus_d_minus_one = (safe_nu - 3.0 + eps).clamp_min(eps)
    return -torch.log1p(safe_kappa) - torch.log1p(nu_minus_d_minus_one)


@torch.no_grad()
def run_pair(
    checkpoint: str | Path,
    frame1: str | Path,
    frame2: str | Path,
    output_npz: str | Path,
    *,
    device: str = "cuda",
    resize_height: int | None = None,
    resize_width: int | None = None,
    center_crop_height: int | None = None,
    center_crop_width: int | None = None,
) -> dict[str, Any]:
    model, checkpoint_payload = load_evifield_era(checkpoint, device=device)
    image1 = load_rgb_image(frame1)
    image2 = load_rgb_image(frame2)
    resize_to = (resize_height, resize_width) if resize_height is not None and resize_width is not None else None
    center_crop = (
        (center_crop_height, center_crop_width)
        if center_crop_height is not None and center_crop_width is not None
        else None
    )
    images, meta = prepare_optical_flow_pair(image1, image2, resize_to=resize_to, center_crop=center_crop)
    images = [image.to(device) for image in images]
    iters_per_scale = 4
    with torch.autocast(device_type="cuda", enabled=False) if str(device).startswith("cuda") else torch.no_grad():
        outputs = model(images, task_type="optical_flow", iters_per_scale=iters_per_scale, use_niw=True)
    niw = outputs["niw"]
    risk_score = analytic_evidence_score(niw["kappa"], niw["nu"])
    output_npz = Path(output_npz)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        pred_field=outputs["pred_field"].detach().cpu().numpy().astype("float32"),
        raw_niw=outputs["raw_niw"].detach().cpu().numpy().astype("float32"),
        kappa=niw["kappa"].detach().cpu().numpy().astype("float32"),
        nu=niw["nu"].detach().cpu().numpy().astype("float32"),
        L=niw["L"].detach().cpu().numpy().astype("float32"),
        Psi=niw["Psi"].detach().cpu().numpy().astype("float32"),
        C_ale=niw["C_ale"].detach().cpu().numpy().astype("float32"),
        C_epi=niw["C_epi"].detach().cpu().numpy().astype("float32"),
        C_total=niw["C_total"].detach().cpu().numpy().astype("float32"),
        U_total=niw["U_total"].detach().cpu().numpy().astype("float32"),
        risk_score=analytic_evidence_score(niw["kappa"], niw["nu"]).detach().cpu().numpy().astype("float32"),
    )
    summary = {
        "checkpoint": str(checkpoint),
        "stage_name": checkpoint_payload.get("stage_name"),
        "iters_per_scale": iters_per_scale,
        "input_meta": meta,
        "output_npz": str(output_npz),
        "pred_field_shape": list(outputs["pred_field"].shape),
        "raw_niw_shape": list(outputs["raw_niw"].shape),
        "C_total_shape": list(niw["C_total"].shape),
        "risk_score_shape": list(risk_score.shape),
        "pred_field_mean": float(outputs["pred_field"].mean().detach().cpu()),
        "pred_field_std": float(outputs["pred_field"].std().detach().cpu()),
        "kappa_mean": float(niw["kappa"].mean().detach().cpu()),
        "nu_mean": float(niw["nu"].mean().detach().cpu()),
        "U_total_mean": float(niw["U_total"].mean().detach().cpu()),
        "risk_score_mean": float(risk_score.mean().detach().cpu()),
    }
    output_npz.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="../checkpoints/evifield_era_optical_flow_latest_S_Mix.pt")
    parser.add_argument("--frame1", required=True)
    parser.add_argument("--frame2", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resize-height", type=int, default=None)
    parser.add_argument("--resize-width", type=int, default=None)
    parser.add_argument("--center-crop-height", type=int, default=None)
    parser.add_argument("--center-crop-width", type=int, default=None)
    args = parser.parse_args()
    summary = run_pair(
        args.checkpoint,
        args.frame1,
        args.frame2,
        args.output,
        device=args.device,
        resize_height=args.resize_height,
        resize_width=args.resize_width,
        center_crop_height=args.center_crop_height,
        center_crop_width=args.center_crop_width,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
