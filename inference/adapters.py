"""Runtime adapters for the packaged DFUN and EviField research models.

The heavy ML dependencies are imported only when direct inference is requested,
so the result-analysis workflow remains usable without PyTorch or SciPy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from io import BytesIO
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DFUN_PACKAGE = ROOT / "dfun_model_package"
EVIFIELD_PACKAGE = ROOT / "system_handoff_evifield_era"


class ModelUnavailableError(RuntimeError):
    """Raised when a model package, checkpoint, or runtime dependency is absent."""


def _load_module(name: str, path: Path) -> ModuleType:
    if not path.is_file():
        raise ModelUnavailableError(f"缺少模型代码文件: {path}")
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ModelUnavailableError(f"无法加载模型代码: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _require_ml_dependencies(*names: str) -> None:
    missing: list[str] = []
    for name in names:
        if importlib.util.find_spec(name) is None:
            missing.append(name)
    if missing:
        packages = ", ".join(missing)
        raise ModelUnavailableError(
            f"缺少模型推理依赖: {packages}。请执行 pip install -e '.[models]'。"
        )


def _auto_device(torch: Any, requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def evifield_arrays_to_frame(
    mean: np.ndarray,
    kappa: np.ndarray,
    nu: np.ndarray,
    base_l: np.ndarray,
) -> pd.DataFrame:
    """Convert EviField arrays to the canonical NIW table without PyTorch."""
    mean = np.asarray(mean)
    kappa = np.asarray(kappa)
    nu = np.asarray(nu)
    base_l = np.asarray(base_l)
    if mean.ndim == 4:
        mean = mean[0]
    if kappa.ndim == 4:
        kappa = kappa[0, 0]
    elif kappa.ndim == 3:
        kappa = kappa[0]
    if nu.ndim == 4:
        nu = nu[0, 0]
    elif nu.ndim == 3:
        nu = nu[0]
    if base_l.ndim == 5:
        base_l = base_l[0]
    if mean.ndim != 3 or mean.shape[0] != 2:
        raise ValueError(f"mean 应为 [2,H,W]，实际为 {mean.shape}")
    height, width = kappa.shape
    if nu.shape != (height, width) or base_l.shape != (height, width, 2, 2):
        raise ValueError("EviField NIW 输出空间尺寸不一致")
    scale = np.sqrt(nu)
    y, x = np.indices((height, width))
    return pd.DataFrame(
        {
            "x": x.reshape(-1),
            "y": y.reshape(-1),
            "mean_1": mean[0].reshape(-1),
            "mean_2": mean[1].reshape(-1),
            "kappa": kappa.reshape(-1),
            "nu": nu.reshape(-1),
            # uqcore expects a Cholesky factor of Psi; EviField defines
            # Psi = nu * (base_l @ base_l.T).
            "l11": (scale * base_l[..., 0, 0]).reshape(-1),
            "l21": (scale * base_l[..., 1, 0]).reshape(-1),
            "l22": (scale * base_l[..., 1, 1]).reshape(-1),
        }
    )


class BaseModelAdapter(ABC):
    model_key: str

    def __init__(self, checkpoint: str | Path, *, device: str = "auto"):
        self.checkpoint = Path(checkpoint)
        self.requested_device = device
        self.device = "cpu"
        self.model: Any | None = None

    @property
    def available(self) -> bool:
        return self.checkpoint.is_file()

    @property
    def loaded(self) -> bool:
        return self.model is not None

    def require_checkpoint(self) -> None:
        if not self.available:
            raise ModelUnavailableError(f"模型参数尚未安装: {self.checkpoint}")

    @abstractmethod
    def load(self) -> None:
        """Load architecture and checkpoint."""

    @abstractmethod
    def predict(self, inputs: Any, **kwargs: Any) -> pd.DataFrame:
        """Run inference and return the canonical analysis-input table."""


class DFUNAdapter(BaseModelAdapter):
    """DFUN MC-Dropout inference producing a long probability table."""

    model_key = "dfun"

    def __init__(
        self,
        checkpoint: str | Path = DFUN_PACKAGE / "checkpoints" / "dfun_gate_fls_exp_test4_model.pth",
        *,
        device: str = "auto",
    ):
        super().__init__(checkpoint, device=device)
        self.model_module: ModuleType | None = None
        self.preprocess_module: ModuleType | None = None

    def load(self) -> None:
        self.require_checkpoint()
        _require_ml_dependencies("torch", "scipy")
        import torch

        code_dir = DFUN_PACKAGE / "dfun"
        self.model_module = _load_module("_sciuq_dfun_model", code_dir / "dfun_model.py")
        self.preprocess_module = _load_module("_sciuq_dfun_preprocess", code_dir / "preprocess.py")
        self.device = _auto_device(torch, self.requested_device)
        self.model = self.model_module.load_dfun_checkpoint(str(self.checkpoint), self.device)

    def _read_input(self, source: Any, sample_index: int) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
        if self.preprocess_module is None:
            raise RuntimeError("DFUN adapter has not been loaded")
        if isinstance(source, (str, Path)):
            return self.preprocess_module.load_sample_npz(source, sample_index=sample_index)
        payload = source.getvalue() if hasattr(source, "getvalue") else source
        if not isinstance(payload, (bytes, bytearray)):
            raise ValueError("DFUN 输入必须是 NPZ 路径或二进制内容")
        data = np.load(BytesIO(payload), allow_pickle=True)
        metadata: dict[str, Any] = {"sample_index": int(sample_index)}
        if "intensity" in data.files:
            intensity = np.asarray(data["intensity"], dtype=np.float32)
            intensity = intensity[sample_index] if intensity.ndim == 2 else intensity.reshape(-1)
        elif "features" in data.files:
            features = np.asarray(data["features"], dtype=np.float32)
            intensity = features[sample_index] if features.ndim == 2 else features.reshape(-1)
        else:
            raise ValueError("NPZ 必须包含 intensity 或 features")
        grid = np.asarray(data["d_spacing"], dtype=np.float32).reshape(-1) if "d_spacing" in data.files else None
        if "labels230" in data.files:
            labels = np.asarray(data["labels230"]).reshape(-1)
            metadata["true_label"] = int(labels[min(sample_index, len(labels) - 1)]) + 1
        elif "space_group" in data.files:
            metadata["true_label"] = int(np.asarray(data["space_group"]).reshape(-1)[0])
        return np.asarray(intensity, dtype=np.float32), grid, metadata

    def predict(
        self,
        inputs: Any,
        *,
        mc_passes: int = 30,
        sample_index: int = 0,
        sample_id: str = "XRD-001",
    ) -> pd.DataFrame:
        if not self.loaded:
            self.load()
        if mc_passes < 2:
            raise ValueError("MC Dropout 推理次数至少为 2")
        import torch

        assert self.model is not None and self.model_module is not None and self.preprocess_module is not None
        intensity, source_grid, metadata = self._read_input(inputs, sample_index)
        true_label = metadata.get("true_label", metadata.get("space_group"))
        raw_xrd, physical, _ = self.preprocess_module.prepare_inputs(intensity, source_grid=source_grid)
        raw_xrd = raw_xrd.to(self.device)
        physical = physical.to(self.device)
        self.model.eval()
        self.model_module.enable_mc_dropout(self.model)
        rows: list[dict[str, Any]] = []
        with torch.inference_mode():
            for pass_index in range(mc_passes):
                logits, gate = self.model(raw_xrd, physical, return_gate=True)
                probabilities = torch.softmax(logits, dim=1)[0].detach().cpu().numpy()
                gate_weight = float(gate[0, 0].detach().cpu())
                for class_index, probability in enumerate(probabilities):
                    row: dict[str, Any] = {
                        "sample_id": sample_id,
                        "pass_id": pass_index + 1,
                        "class_label": class_index + 1,
                        "probability": float(probability),
                        "gate_weight": gate_weight,
                    }
                    if true_label is not None:
                        row["true_label"] = int(true_label)
                    rows.append(row)
        return pd.DataFrame(rows)


class EviFieldAdapter(BaseModelAdapter):
    """EviField optical-flow NIW inference producing a pixel parameter table."""

    model_key = "evifield"

    def __init__(
        self,
        checkpoint: str | Path = EVIFIELD_PACKAGE / "checkpoints" / "evifield_era_optical_flow_latest_S_Mix.pt",
        *,
        device: str = "auto",
    ):
        super().__init__(checkpoint, device=device)
        self.preprocess_module: ModuleType | None = None
        self.checkpoint_payload: dict[str, Any] = {}

    def load(self) -> None:
        self.require_checkpoint()
        _require_ml_dependencies("torch", "PIL")
        import torch

        code_dir = EVIFIELD_PACKAGE / "code"
        previous_evidential = sys.modules.get("evidential")
        evidential = _load_module("evidential", code_dir / "evidential.py")
        try:
            model_module = _load_module("_sciuq_evifield_model", code_dir / "model.py")
        finally:
            if previous_evidential is not None:
                sys.modules["evidential"] = previous_evidential
            else:
                sys.modules.pop("evidential", None)
        self.preprocess_module = _load_module("_sciuq_evifield_preprocess", code_dir / "preprocess.py")
        payload = torch.load(self.checkpoint, map_location="cpu")
        runtime = payload.get("runtime_config_effective") or payload.get("config", {}).get("runtime", {})
        model = model_module.UniVecEDL(
            evidential_mode=runtime.get("evidential_mode", "niw"),
            flow_support_aggregation=runtime.get("flow_support_aggregation", "warp"),
            niw_r=float(runtime.get("niw_r", 1.0)),
            niw_eps=float(runtime.get("niw_eps", 1e-6)),
            niw_nu_max=runtime.get("niw_nu_max"),
            niw_nu_min=float(runtime.get("niw_nu_min", 3.1)),
            use_risk_projector=bool(runtime.get("use_risk_projector", False)),
            risk_projector_features=runtime.get("risk_projector_features", "covariance"),
        )
        state = payload["model"]
        if any(key.startswith("module.") for key in state):
            state = {key.removeprefix("module."): value for key, value in state.items()}
        load_result = model.load_state_dict(state, strict=False)
        missing = [key for key in load_result.missing_keys if not key.startswith("final_gaussian_head.")]
        if missing or load_result.unexpected_keys:
            raise ModelUnavailableError(
                f"EviField 参数不匹配: missing={missing}, unexpected={list(load_result.unexpected_keys)}"
            )
        self.device = _auto_device(torch, self.requested_device)
        self.model = model.to(self.device).eval()
        self.checkpoint_payload = payload
        self._evidential_module = evidential

    @staticmethod
    def _image_tensor(source: Any) -> Any:
        import torch
        from PIL import Image

        if isinstance(source, np.ndarray):
            array = source
        else:
            if isinstance(source, (str, Path)):
                image = Image.open(source)
            else:
                payload = source.getvalue() if hasattr(source, "getvalue") else source
                if not isinstance(payload, (bytes, bytearray)):
                    raise ValueError("图像输入必须是路径、数组或二进制内容")
                image = Image.open(BytesIO(payload))
            array = np.array(image.convert("RGB"), copy=True)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("EviField 输入图像必须是 RGB 三通道")
        return torch.from_numpy(array).permute(2, 0, 1).contiguous().float()

    def predict(
        self,
        inputs: Any,
        *,
        resize_to: tuple[int, int] | None = None,
    ) -> pd.DataFrame:
        if not self.loaded:
            self.load()
        if not isinstance(inputs, (tuple, list)) or len(inputs) != 2:
            raise ValueError("EviField 需要按顺序提供参考帧和支持帧")
        import torch

        assert self.model is not None and self.preprocess_module is not None
        frame1 = self._image_tensor(inputs[0])
        frame2 = self._image_tensor(inputs[1])
        images, _ = self.preprocess_module.prepare_optical_flow_pair(frame1, frame2, resize_to=resize_to)
        images = [image.to(self.device) for image in images]
        with torch.inference_mode():
            outputs = self.model(images, task_type="optical_flow", iters_per_scale=4, use_niw=True)
        niw = outputs["niw"]
        return evifield_arrays_to_frame(
            niw["mean"].detach().cpu().numpy(),
            niw["kappa"].detach().cpu().numpy(),
            niw["nu"].detach().cpu().numpy(),
            niw["L"].detach().cpu().numpy(),
        )


def model_registry() -> list[dict[str, Any]]:
    """Return UI-safe package status without importing ML dependencies."""
    return [
        {
            "name": "DFUN",
            "checkpoint": DFUNAdapter().checkpoint,
            "task": "空间群分类 / MC Dropout",
            "output": "T×230 类别概率",
            "requirements": "NPZ 衍射曲线 · 5000点重采样 · 45维峰值特征",
        },
        {
            "name": "EviField",
            "checkpoint": EviFieldAdapter().checkpoint,
            "task": "双帧光流 / 二维 NIW",
            "output": "m, κ, ν, L 与全协方差",
            "requirements": "两张RGB图像 · 自动归一化 · 像素级证据输出",
        },
    ]
