#!/usr/bin/env python3
"""Minimal EviField-ERA LTEM inference example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inference_minimal import analytic_evidence_score, load_evifield_era
from preprocess import load_ltem_image, prepare_ltem_triplet


@torch.no_grad()
def run_ltem(
    checkpoint: str | Path,
    under: str | Path,
    infocus: str | Path,
    over: str | Path,
    output_npz: str | Path,
    *,
    device: str = "cuda",
    formal_val_preprocess: bool = True,
) -> dict[str, Any]:
    model, checkpoint_payload = load_evifield_era(checkpoint, device=device)
    resize_to = (256, 256) if formal_val_preprocess else None
    center_crop = (224, 224) if formal_val_preprocess else None
    images, meta = prepare_ltem_triplet(
        load_ltem_image(under),
        load_ltem_image(infocus),
        load_ltem_image(over),
        resize_to=resize_to,
        center_crop=center_crop,
    )
    images = [image.to(device) for image in images]
    outputs = model(images, task_type="ltem", iters_per_scale=4, use_niw=True)
    niw = outputs["niw"]
    risk_score = outputs.get("risk_score_crp")
    if risk_score is None:
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
        risk_score=risk_score.detach().cpu().numpy().astype("float32"),
    )
    summary = {
        "checkpoint": str(checkpoint),
        "stage_name": checkpoint_payload.get("stage_name"),
        "task_type": "ltem",
        "iters_per_scale": 4,
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
    parser.add_argument("--checkpoint", default="../checkpoints/evifield_era_ltem_latest_FT.pt")
    parser.add_argument("--under", required=True)
    parser.add_argument("--infocus", required=True)
    parser.add_argument("--over", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--no-formal-val-preprocess", action="store_true")
    args = parser.parse_args()
    summary = run_ltem(
        args.checkpoint,
        args.under,
        args.infocus,
        args.over,
        args.output,
        device=args.device,
        formal_val_preprocess=not args.no_formal_val_preprocess,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
