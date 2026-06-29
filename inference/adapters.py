"""Stable interfaces reserved for the paper-model checkpoints supplied later."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any


class ModelUnavailableError(RuntimeError):
    """Raised when inference is requested before a checkpoint is installed."""


class BaseModelAdapter(ABC):
    model_key: str

    def __init__(self, checkpoint: str | Path):
        self.checkpoint = Path(checkpoint)

    @property
    def available(self) -> bool:
        return self.checkpoint.is_file()

    def require_checkpoint(self) -> None:
        if not self.available:
            raise ModelUnavailableError(
                f"模型参数尚未安装: {self.checkpoint}。请按 models/README.md 放置参数文件。"
            )

    @abstractmethod
    def load(self) -> None:
        """Load model architecture and checkpoint."""

    @abstractmethod
    def predict(self, inputs: Any) -> Any:
        """Run inference and return the canonical analysis input."""


class DFUNAdapter(BaseModelAdapter):
    """Placeholder for DFUN MC-Dropout probability inference."""

    model_key = "dfun"

    def load(self) -> None:
        self.require_checkpoint()
        raise NotImplementedError("收到 DFUN 架构代码和参数文件后实现加载映射")

    def predict(self, inputs: Any) -> Any:
        self.require_checkpoint()
        raise NotImplementedError("收到 DFUN 预处理配置后实现 T 次随机前向传播")


class EviFieldAdapter(BaseModelAdapter):
    """Placeholder for EviField per-pixel NIW inference."""

    model_key = "evifield"

    def load(self) -> None:
        self.require_checkpoint()
        raise NotImplementedError("收到 EviField 架构代码和参数文件后实现加载映射")

    def predict(self, inputs: Any) -> Any:
        self.require_checkpoint()
        raise NotImplementedError("收到 EviField 预处理配置后输出 NIW 参数")

