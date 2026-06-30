from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from dfun_model import enable_mc_dropout, load_dfun_checkpoint
from preprocess import load_sample_npz, prepare_inputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal DFUN inference example.")
    parser.add_argument("--checkpoint", default=str(PACKAGE_ROOT / "checkpoints" / "dfun_gate_fls_exp_test4_model.pth"))
    parser.add_argument("--sample", default=str(PACKAGE_ROOT / "samples" / "opxrd_public_sample.npz"))
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--mc-dropout", action="store_true")
    parser.add_argument("--mc-passes", type=int, default=30)
    parser.add_argument("--output", default="")
    return parser.parse_args()


def predict(
    checkpoint: str | Path,
    sample: str | Path,
    sample_index: int = 0,
    device: str = "cpu",
    top_k: int = 5,
    mc_dropout: bool = False,
    mc_passes: int = 30,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    model = load_dfun_checkpoint(str(checkpoint), device_obj)

    intensity, source_grid, sample_metadata = load_sample_npz(sample, sample_index=sample_index)
    raw_xrd, physical_features, d_grid = prepare_inputs(intensity, source_grid=source_grid)
    raw_xrd = raw_xrd.to(device_obj)
    physical_features = physical_features.to(device_obj)

    dropout_modules_enabled: list[str] = []
    with torch.no_grad():
        if mc_dropout:
            model.eval()
            dropout_modules_enabled = enable_mc_dropout(model)
            probs_all = []
            gates_all = []
            logits_last = None
            for _ in range(mc_passes):
                logits, gate = model(raw_xrd, physical_features, return_gate=True)
                logits_last = logits
                probs_all.append(F.softmax(logits, dim=1).cpu().numpy())
                gates_all.append(gate.cpu().numpy())
            probabilities = np.mean(np.concatenate(probs_all, axis=0), axis=0)
            gate_weight = float(np.mean(np.concatenate(gates_all, axis=0)))
            logits_np = logits_last.cpu().numpy().reshape(-1)
            uncertainty_source = "mc_dropout"
        else:
            model.eval()
            logits, gate = model(raw_xrd, physical_features, return_gate=True)
            probabilities = F.softmax(logits, dim=1).cpu().numpy().reshape(-1)
            logits_np = logits.cpu().numpy().reshape(-1)
            gate_weight = float(gate.cpu().numpy().reshape(-1)[0])
            uncertainty_source = "single_softmax"

    top_k = min(int(top_k), probabilities.shape[0])
    order = np.argsort(probabilities)[::-1][:top_k]
    entropy = float(-(probabilities * np.log(probabilities + 1e-12)).sum())

    result = {
        "sample_metadata": sample_metadata,
        "checkpoint": str(checkpoint),
        "input_shapes": {
            "raw_xrd": list(raw_xrd.shape),
            "physical_features": list(physical_features.shape),
            "logits": list(logits_np.shape),
            "probabilities": list(probabilities.shape),
        },
        "uncertainty_source": uncertainty_source,
        "mc_passes": int(mc_passes) if mc_dropout else 1,
        "dropout_modules_enabled": dropout_modules_enabled,
        "gate_weight": gate_weight,
        "gate_definition": "larger gate_weight means stronger reliance on the CNN/raw-XRD branch",
        "predicted_label_index": int(order[0]),
        "predicted_space_group": int(order[0] + 1),
        "top_k": [
            {
                "rank": int(rank + 1),
                "label_index": int(label),
                "space_group": int(label + 1),
                "probability": float(probabilities[label]),
            }
            for rank, label in enumerate(order)
        ],
        "predictive_entropy": entropy,
        "probabilities": [float(x) for x in probabilities.tolist()],
    }
    return result


def main() -> None:
    args = parse_args()
    result = predict(
        checkpoint=args.checkpoint,
        sample=args.sample,
        sample_index=args.sample_index,
        device=args.device,
        top_k=args.top_k,
        mc_dropout=args.mc_dropout,
        mc_passes=args.mc_passes,
    )
    text = json.dumps(result, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
