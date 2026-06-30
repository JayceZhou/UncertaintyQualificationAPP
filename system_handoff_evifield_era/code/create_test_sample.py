#!/usr/bin/env python3
"""Create a deterministic synthetic two-frame RGB test sample."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    out_dir = Path(__file__).resolve().parents[1] / "test_sample"
    out_dir.mkdir(parents=True, exist_ok=True)
    height, width = 64, 96
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    base = np.stack(
        [
            np.broadcast_to(x, (height, width)),
            np.broadcast_to(y, (height, width)),
            0.5 + 0.5 * np.sin(2.0 * np.pi * (x + y)),
        ],
        axis=-1,
    )
    shift = 3
    frame1 = np.clip(base * 255.0, 0, 255).astype(np.uint8)
    frame2 = np.roll(frame1, shift=shift, axis=1)
    Image.fromarray(frame1).save(out_dir / "frame1.png")
    Image.fromarray(frame2).save(out_dir / "frame2.png")
    (out_dir / "README.md").write_text(
        "Deterministic synthetic optical-flow smoke-test pair. frame2 is frame1 shifted by 3 pixels along +x.\n",
        encoding="utf-8",
    )
    print(out_dir)


if __name__ == "__main__":
    main()
