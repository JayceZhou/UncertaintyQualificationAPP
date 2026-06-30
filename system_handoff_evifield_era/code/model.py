"""UniVec-EDL model for dense 2D vector-field estimation with final-scale evidential uncertainty."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from evidential import niw_covariance_risk_features, unpack_gaussian_full_cov_2d, unpack_nig_diag_2d, unpack_niw_2d

__all__ = [
    "SharedPyramidEncoder",
    "OmniFusionTransformer",
    "ConvGRUCell",
    "DecoderStage",
    "FinalNIGHead2D",
    "FinalGaussianHead2D",
    "FinalNIWHead2D",
    "FinalMvDERHead2D",
    "CovarianceRiskProjector",
    "UniVecEDL",
]


EPS = 1e-6


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _adapt_image_channels(image: Tensor) -> Tensor:
    """Normalize image channels to a 3-channel encoder contract."""
    _require(image.ndim == 4, f"Expected image tensor with shape [B, C, H, W], got {tuple(image.shape)}.")
    channels = image.shape[1]
    if channels == 3:
        return image
    if channels == 1:
        return image.repeat(1, 3, 1, 1)
    raise ValueError(
        "UniVecEDL currently supports 1-channel or 3-channel images. "
        f"Received C={channels}."
    )


def _validate_images(images: Sequence[Tensor], task_type: str) -> Tuple[int, int, int]:
    if task_type not in {"optical_flow", "ltem"}:
        raise ValueError(f"Unsupported task_type='{task_type}'. Expected 'optical_flow' or 'ltem'.")
    expected_images = 2 if task_type == "optical_flow" else 3
    if len(images) != expected_images:
        raise ValueError(
            f"task_type='{task_type}' expects {expected_images} input images, received {len(images)}."
        )
    reference_shape = tuple(images[0].shape)
    _require(len(reference_shape) == 4, f"Expected [B, C, H, W], got {reference_shape}.")
    batch, _, height, width = reference_shape
    for index, image in enumerate(images):
        _require(
            image.ndim == 4,
            f"Image at index {index} must have shape [B, C, H, W], got {tuple(image.shape)}.",
        )
        _require(
            image.shape[0] == batch and image.shape[-2:] == (height, width),
            "All images must share batch size and spatial dimensions. "
            f"Reference shape is {reference_shape}, received {tuple(image.shape)} at index {index}.",
        )
        _require(
            image.device == images[0].device and image.dtype == images[0].dtype,
            "All images must share the same device and dtype.",
        )
    return batch, height, width


def _next_multiple(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def _pad_image(image: Tensor, target_height: int, target_width: int) -> Tensor:
    pad_h = target_height - image.shape[-2]
    pad_w = target_width - image.shape[-1]
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), mode="replicate")


def _crop_spatial(tensor: Tensor, target_hw: Tuple[int, int]) -> Tensor:
    target_h, target_w = target_hw
    return tensor[..., :target_h, :target_w]


def _successive_downsample_sizes(height: int, width: int, steps: int) -> List[Tuple[int, int]]:
    sizes: List[Tuple[int, int]] = []
    current_h, current_w = height, width
    for _ in range(steps):
        current_h = math.ceil(current_h / 2)
        current_w = math.ceil(current_w / 2)
        sizes.append((current_h, current_w))
    return sizes


def _cast_feature_scales_to_float32(scales: Sequence[Tensor]) -> List[Tensor]:
    return [scale.float() for scale in scales]


def _cast_nested_feature_scales_to_float32(scales: Sequence[Sequence[Tensor]]) -> List[List[Tensor]]:
    return [[scale.float() for scale in level] for level in scales]


def _resize_vector_field(field: Tensor, size: Tuple[int, int]) -> Tensor:
    """Resize [dx, dy] field while preserving displacement magnitudes."""
    _require(field.ndim == 4 and field.shape[1] == 2, f"Expected field [B, 2, H, W], got {tuple(field.shape)}.")
    target_h, target_w = size
    source_h, source_w = field.shape[-2:]
    if (source_h, source_w) == (target_h, target_w):
        return field
    resized = F.interpolate(field, size=size, mode="bilinear", align_corners=False)
    scale_x = target_w / max(source_w, 1)
    scale_y = target_h / max(source_h, 1)
    resized_x = resized[:, 0:1] * scale_x
    resized_y = resized[:, 1:2] * scale_y
    return torch.cat([resized_x, resized_y], dim=1)


def _resize_mask(mask: Tensor, size: Tuple[int, int]) -> Tensor:
    _require(mask.ndim == 4 and mask.shape[1] == 1, f"Expected mask [B, 1, H, W], got {tuple(mask.shape)}.")
    if mask.shape[-2:] == size:
        return mask
    return F.interpolate(mask, size=size, mode="nearest")


def _base_grid(height: int, width: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    y = (torch.arange(height, device=device, dtype=dtype) + 0.5) * (2.0 / max(height, 1)) - 1.0
    x = (torch.arange(width, device=device, dtype=dtype) + 0.5) * (2.0 / max(width, 1)) - 1.0
    grid_y, grid_x = torch.meshgrid(y, x, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)


def warp_features(features: Tensor, flow: Tensor) -> Tensor:
    """Warp support features with [dx, dy] flow using align_corners=False."""
    _require(
        features.ndim == 4 and flow.ndim == 4,
        f"Expected 4D tensors for warping, got features={tuple(features.shape)}, flow={tuple(flow.shape)}.",
    )
    _require(features.shape[0] == flow.shape[0], "Features and flow must share batch size.")
    _require(flow.shape[1] == 2, f"Expected flow channels [dx, dy], got {flow.shape[1]}.")
    if features.shape[-2:] != flow.shape[-2:]:
        flow = _resize_vector_field(flow, features.shape[-2:])
    height, width = features.shape[-2:]
    grid = _base_grid(height, width, features.device, features.dtype).expand(features.shape[0], -1, -1, -1)
    offset_x = 2.0 * flow[:, 0] / max(width, 1)
    offset_y = 2.0 * flow[:, 1] / max(height, 1)
    warped_grid = grid + torch.stack([offset_x, offset_y], dim=-1)
    return F.grid_sample(features, warped_grid, mode="bilinear", padding_mode="border", align_corners=False)


def _build_2d_sincos_position_embedding(
    height: int,
    width: int,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    _require(dim % 4 == 0, f"Positional embedding dim must be divisible by 4, got {dim}.")
    half_dim = dim // 2
    quarter_dim = dim // 4
    omega = torch.arange(quarter_dim, device=device, dtype=dtype)
    omega = 1.0 / (10000.0 ** (omega / max(quarter_dim - 1, 1)))

    y_pos = torch.arange(height, device=device, dtype=dtype).unsqueeze(1) * omega.unsqueeze(0)
    x_pos = torch.arange(width, device=device, dtype=dtype).unsqueeze(1) * omega.unsqueeze(0)

    emb_y = torch.cat([torch.sin(y_pos), torch.cos(y_pos)], dim=1)
    emb_x = torch.cat([torch.sin(x_pos), torch.cos(x_pos)], dim=1)

    pos_y = emb_y[:, None, :].expand(height, width, half_dim)
    pos_x = emb_x[None, :, :].expand(height, width, half_dim)
    return torch.cat([pos_y, pos_x], dim=-1).reshape(1, height * width, dim)


def _resolve_evidential_mode(use_niw: bool, evidential_mode: str) -> str:
    if evidential_mode not in {"none", "nig", "niw", "mvder", "gaussian"}:
        raise ValueError(f"Unsupported evidential_mode='{evidential_mode}'. Expected 'none', 'nig', 'niw', 'mvder', or 'gaussian'.")
    if not use_niw:
        return "none"
    return evidential_mode


def _covariance_to_legacy_layout(covariance: Tensor) -> Tensor:
    _require(
        covariance.ndim == 5 and covariance.shape[-2:] == (2, 2),
        f"Expected covariance [B, H, W, 2, 2], got {tuple(covariance.shape)}.",
    )
    return covariance.permute(0, 3, 4, 1, 2).contiguous()


def _diag_to_legacy_covariance(diagonal: Tensor) -> Tensor:
    _require(diagonal.ndim == 4 and diagonal.shape[1] == 2, f"Expected diagonal [B, 2, H, W], got {tuple(diagonal.shape)}.")
    batch, _, height, width = diagonal.shape
    covariance = diagonal.new_zeros((batch, 2, 2, height, width))
    covariance[:, 0, 0] = diagonal[:, 0]
    covariance[:, 1, 1] = diagonal[:, 1]
    return covariance


class ConvNormAct(nn.Module):
    """Conv-BN-ReLU helper block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    """Lightweight residual block used by the shared encoder."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: Tensor) -> Tensor:
        residual = self.shortcut(x)
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.activation(out + residual)


class SharedPyramidEncoder(nn.Module):
    """Shared-weight encoder producing 1/4, 1/8, and 1/16 feature maps."""

    def __init__(
        self,
        in_channels: int = 3,
        scale_channels: Tuple[int, int, int] = (128, 256, 512),
        bottleneck_dim: int = 1024,
    ) -> None:
        super().__init__()
        c4, c8, c16 = scale_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            ResidualBlock(64, c4, stride=2),
            ResidualBlock(c4, c4),
        )
        self.layer8 = nn.Sequential(
            ResidualBlock(c4, c8, stride=2),
            ResidualBlock(c8, c8),
        )
        self.layer16 = nn.Sequential(
            ResidualBlock(c8, c16, stride=2),
            ResidualBlock(c16, c16),
        )
        self.bottleneck_proj = nn.Conv2d(c16, bottleneck_dim, kernel_size=1)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        stem = self.stem(x)  # [B, 64, H/2, W/2]
        scale4 = self.layer4(stem)  # [B, 128, H/4, W/4]
        scale8 = self.layer8(scale4)  # [B, 256, H/8, W/8]
        scale16 = self.layer16(scale8)  # [B, 512, H/16, W/16]
        bottleneck = self.bottleneck_proj(scale16)  # [B, 1024, H/16, W/16]
        return {
            "scale_4": scale4,
            "scale_8": scale8,
            "scale_16": scale16,
            "bottleneck": bottleneck,
        }


class OmniFusionTransformer(nn.Module):
    """Cross-attention fusion from a reference bottleneck into supporting bottlenecks."""

    def __init__(self, embed_dim: int = 1024, num_heads: int = 8, mlp_ratio: int = 2) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * mlp_ratio),
            nn.GELU(),
            nn.Linear(embed_dim * mlp_ratio, embed_dim),
        )

    def forward(self, reference: Tensor, supports: Sequence[Tensor]) -> Tensor:
        _require(len(supports) > 0, "OmniFusionTransformer requires at least one supporting feature map.")
        batch, channels, height, width = reference.shape
        reference_tokens = reference.flatten(2).transpose(1, 2)
        support_tokens = torch.cat([support.flatten(2).transpose(1, 2) for support in supports], dim=1)

        pos = _build_2d_sincos_position_embedding(height, width, channels, reference.device, reference.dtype)
        repeated_pos = pos.repeat(1, len(supports), 1)

        query = reference_tokens + pos
        key_value = support_tokens + repeated_pos

        attn_out, _ = self.attention(query, key_value, key_value, need_weights=False)
        fused = self.norm1(reference_tokens + attn_out)
        fused = self.norm2(fused + self.mlp(fused))
        return fused.transpose(1, 2).reshape(batch, channels, height, width)


class ConvGRUCell(nn.Module):
    """Convolutional GRU cell for recurrent feature updates."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gates = nn.Conv2d(input_dim + hidden_dim, 2 * hidden_dim, kernel_size=3, padding=1)
        self.candidate = nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=3, padding=1)

    def forward(self, x: Tensor, hidden: Tensor) -> Tensor:
        if hidden is None:
            hidden = torch.zeros(
                x.shape[0],
                self.hidden_dim,
                x.shape[-2],
                x.shape[-1],
                device=x.device,
                dtype=x.dtype,
            )
        combined = torch.cat([x, hidden], dim=1)
        update_gate, reset_gate = self.gates(combined).chunk(2, dim=1)
        update_gate = torch.sigmoid(update_gate)
        reset_gate = torch.sigmoid(reset_gate)
        candidate = torch.tanh(self.candidate(torch.cat([x, hidden * reset_gate], dim=1)))
        return (1.0 - update_gate) * hidden + update_gate * candidate


class FinalNIWHead2D(nn.Module):
    """Predict raw 2D NIW parameters with mean channels anchored to the decoder field."""

    def __init__(
        self,
        hidden_dim: int,
        raw_channels: int = 6,
    ) -> None:
        super().__init__()
        _require(raw_channels in {6, 7}, f"raw_channels must be 6 or 7, got {raw_channels}.")
        self.raw_channels = raw_channels
        self.head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, raw_channels, kernel_size=3, padding=1),
        )

    def forward(self, hidden: Tensor, base_field: Tensor) -> Tensor:
        raw = self.head(hidden)
        raw = raw.clone()
        raw[:, 0:2] = raw[:, 0:2] + base_field
        return raw


class FinalMvDERHead2D(FinalNIWHead2D):
    """Direct dense multivariate DER/NIW head with independently predicted nu."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__(hidden_dim, raw_channels=7)


class FinalNIGHead2D(nn.Module):
    """Predict independent per-component NIG parameters anchored to the decoder field."""

    def __init__(
        self,
        hidden_dim: int,
    ) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, 8, kernel_size=3, padding=1),
        )

    def forward(self, hidden: Tensor, base_field: Tensor) -> Tensor:
        raw = self.head(hidden)
        raw = raw.clone()
        raw[:, 0:2] = raw[:, 0:2] + base_field
        return raw


class FinalGaussianHead2D(nn.Module):
    """Predict a full 2x2 Gaussian covariance anchored to the decoder field."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, 5, kernel_size=3, padding=1),
        )

    def forward(self, hidden: Tensor, base_field: Tensor) -> Tensor:
        raw = self.head(hidden)
        raw = raw.clone()
        raw[:, 0:2] = raw[:, 0:2] + base_field
        return raw


class CovarianceRiskProjector(nn.Module):
    """Small per-pixel projector from NIW covariance attributes to scalar risk."""

    def __init__(self, feature_mode: str = "covariance", hidden_dim: int = 16) -> None:
        super().__init__()
        normalized = feature_mode.lower().replace("-", "_")
        feature_dims = {
            "covariance": 10,
            "full": 10,
            "no_coupling": 7,
            "strict_no_coupling": 7,
            "evidence_only": 2,
            "evidence_trace": 3,
            "monotonic_evidence": 2,
        }
        _require(normalized in feature_dims, f"Unsupported risk projector features '{feature_mode}'.")
        self.feature_mode = normalized
        if self.feature_mode == "monotonic_evidence":
            self.monotonic_bias = nn.Parameter(torch.zeros(1))
            self.monotonic_weight_raw = nn.Parameter(torch.zeros(2))
            self.net = None
        else:
            self.net = nn.Sequential(
                nn.Conv2d(feature_dims[normalized], hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(hidden_dim, 1, kernel_size=1),
                nn.Softplus(),
            )

    def forward(self, niw_outputs: Mapping[str, Tensor]) -> Tensor:
        features = niw_covariance_risk_features(niw_outputs, feature_set=self.feature_mode)
        if self.feature_mode == "monotonic_evidence":
            weights = F.softplus(self.monotonic_weight_raw).view(1, 2, 1, 1)
            score = F.softplus(self.monotonic_bias.view(1, 1, 1, 1) - (features * weights).sum(dim=1, keepdim=True))
        else:
            _require(self.net is not None, "Risk projector network is not initialized.")
            score = self.net(features)
        return torch.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


class DecoderStage(nn.Module):
    """Single recurrent decoding stage at one spatial scale."""

    def __init__(self, ref_channels: int, support_channels: int, hidden_dim: int) -> None:
        super().__init__()
        self.ref_proj = ConvNormAct(ref_channels, hidden_dim, kernel_size=3)
        self.support_proj = ConvNormAct(support_channels, hidden_dim, kernel_size=3)
        self.input_proj = ConvNormAct(hidden_dim * 2 + 2, hidden_dim, kernel_size=3)
        self.gru = ConvGRUCell(hidden_dim, hidden_dim)
        self.delta_head = nn.Sequential(
            ConvNormAct(hidden_dim, hidden_dim, kernel_size=3),
            nn.Conv2d(hidden_dim, 2, kernel_size=3, padding=1),
        )

    def forward(
        self,
        reference: Tensor,
        support: Tensor,
        hidden: Tensor,
        field: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        ref_context = self.ref_proj(reference)
        support_context = self.support_proj(support)
        gru_input = self.input_proj(torch.cat([ref_context, support_context, field], dim=1))
        hidden = self.gru(gru_input, hidden)
        delta_v = self.delta_head(hidden)
        field = field + delta_v
        return hidden, field


class FullResolutionRefiner(nn.Module):
    """Learnable upsampling block used to synthesize 1/1 features from 1/4 features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.refine = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3),
            ConvNormAct(out_channels, out_channels, kernel_size=3),
        )

    def forward(self, features: Tensor, target_size: Tuple[int, int]) -> Tensor:
        upsampled = F.interpolate(features, size=target_size, mode="bilinear", align_corners=False)
        return self.refine(upsampled)


class UniVecEDL(nn.Module):
    """Unified dense vector-field estimator with deterministic decoding and final-scale NIW outputs."""

    def __init__(
        self,
        bottleneck_dim: int = 1024,
        decoder_hidden_dims: Tuple[int, int, int, int] = (256, 192, 128, 96),
        encoder_channels: Tuple[int, int, int] = (128, 256, 512),
        evidential_mode: str = "none",
        flow_support_aggregation: str = "warp",
        niw_r: float = 1.0,
        niw_eps: float = EPS,
        niw_nu_max: Optional[float] = None,
        niw_nu_min: float = 3.01,
        use_risk_projector: bool = False,
        risk_projector_features: str = "covariance",
        gaussian_use_base_mean: bool = False,
        gaussian_use_niw_mean: bool = False,
    ) -> None:
        super().__init__()
        self.evidential_mode = evidential_mode
        _require(
            not (gaussian_use_base_mean and gaussian_use_niw_mean),
            "gaussian_use_base_mean and gaussian_use_niw_mean are mutually exclusive.",
        )
        _require(
            flow_support_aggregation in {"warp", "nonwarp"},
            f"flow_support_aggregation must be 'warp' or 'nonwarp', got {flow_support_aggregation!r}.",
        )
        self.flow_support_aggregation = flow_support_aggregation
        self.niw_r = niw_r
        self.niw_eps = niw_eps
        self.niw_nu_max = niw_nu_max
        self.niw_nu_min = niw_nu_min
        self.gaussian_use_base_mean = bool(gaussian_use_base_mean)
        self.gaussian_use_niw_mean = bool(gaussian_use_niw_mean)
        self.use_risk_projector = bool(use_risk_projector)
        self.risk_projector_features = risk_projector_features
        self.encoder = SharedPyramidEncoder(
            in_channels=3,
            scale_channels=encoder_channels,
            bottleneck_dim=bottleneck_dim,
        )
        self.fusion = OmniFusionTransformer(embed_dim=bottleneck_dim, num_heads=8, mlp_ratio=2)
        self.full_res_refiner = FullResolutionRefiner(encoder_channels[0], decoder_hidden_dims[-1])

        stage_channels = [bottleneck_dim, encoder_channels[1], encoder_channels[0], decoder_hidden_dims[-1]]
        self.decoder_stages = nn.ModuleList(
            [
                DecoderStage(stage_channels[0], stage_channels[0], decoder_hidden_dims[0]),
                DecoderStage(stage_channels[1], stage_channels[1], decoder_hidden_dims[1]),
                DecoderStage(stage_channels[2], stage_channels[2], decoder_hidden_dims[2]),
                DecoderStage(stage_channels[3], stage_channels[3], decoder_hidden_dims[3]),
            ]
        )

        self.hidden_adapters = nn.ModuleList(
            [
                nn.Conv2d(decoder_hidden_dims[0], decoder_hidden_dims[1], kernel_size=1),
                nn.Conv2d(decoder_hidden_dims[1], decoder_hidden_dims[2], kernel_size=1),
                nn.Conv2d(decoder_hidden_dims[2], decoder_hidden_dims[3], kernel_size=1),
            ]
        )
        self.final_niw_head = FinalNIWHead2D(decoder_hidden_dims[-1])
        self.final_mvder_head = FinalMvDERHead2D(decoder_hidden_dims[-1])
        self.final_nig_head = FinalNIGHead2D(decoder_hidden_dims[-1])
        self.final_gaussian_head = FinalGaussianHead2D(decoder_hidden_dims[-1])
        self.risk_projector = (
            CovarianceRiskProjector(feature_mode=risk_projector_features)
            if self.use_risk_projector
            else None
        )

    def _crop_encoder_features(
        self,
        encoded: Dict[str, Tensor],
        target_sizes: Dict[str, Tuple[int, int]],
    ) -> Dict[str, Tensor]:
        return {
            "scale_4": _crop_spatial(encoded["scale_4"], target_sizes["scale_4"]),
            "scale_8": _crop_spatial(encoded["scale_8"], target_sizes["scale_8"]),
            "scale_16": _crop_spatial(encoded["scale_16"], target_sizes["scale_16"]),
            "bottleneck": _crop_spatial(encoded["bottleneck"], target_sizes["scale_16"]),
        }

    def _encode_images(self, images: Sequence[Tensor], target_sizes: Dict[str, Tuple[int, int]]) -> List[Dict[str, Tensor]]:
        encoded_images: List[Dict[str, Tensor]] = []
        for image in images:
            encoded_images.append(self._crop_encoder_features(self.encoder(image), target_sizes))
        return encoded_images

    def _aggregate_support(self, supports: Sequence[Tensor], flow: Tensor, task_type: str) -> Tensor:
        warped_supports: List[Tensor] = []
        for support in supports:
            if task_type == "optical_flow" and self.flow_support_aggregation == "warp":
                warped_supports.append(warp_features(support, flow))
            else:
                warped_supports.append(support)
        return torch.stack(warped_supports, dim=0).mean(dim=0)

    def _prepare_scale_features(
        self,
        encoded_images: Sequence[Dict[str, Tensor]],
        full_size: Tuple[int, int],
    ) -> Tuple[List[Tensor], List[List[Tensor]]]:
        reference = encoded_images[0]
        supports = encoded_images[1:]

        fused_bottleneck = self.fusion(reference["bottleneck"], [support["bottleneck"] for support in supports])
        reference_scales = [
            fused_bottleneck,
            reference["scale_8"],
            reference["scale_4"],
            self.full_res_refiner(reference["scale_4"], full_size),
        ]

        support_scales = [
            [support["bottleneck"] for support in supports],
            [support["scale_8"] for support in supports],
            [support["scale_4"] for support in supports],
            [self.full_res_refiner(support["scale_4"], full_size) for support in supports],
        ]
        return reference_scales, support_scales

    def forward(
        self,
        images: List[Tensor],
        task_type: str = "optical_flow",
        iters_per_scale: int = 4,
        use_niw: bool = True,
        fp32_feature_prep: bool = False,
        sef_grad_mode: str = "normal",
    ) -> Dict[str, Any]:
        """Run UniVec-EDL on optical flow or LTEM inputs."""
        batch, original_h, original_w = _validate_images(images, task_type)
        _require(iters_per_scale > 0, f"iters_per_scale must be positive, got {iters_per_scale}.")
        _require(sef_grad_mode in {"normal", "head_only"}, f"Unsupported sef_grad_mode='{sef_grad_mode}'.")

        adapted_images = [_adapt_image_channels(image) for image in images]
        padded_h = _next_multiple(original_h, 16)
        padded_w = _next_multiple(original_w, 16)
        padded_images = [_pad_image(image, padded_h, padded_w) for image in adapted_images]

        downsample_sizes = _successive_downsample_sizes(original_h, original_w, steps=4)
        target_sizes = {
            "scale_4": downsample_sizes[1],
            "scale_8": downsample_sizes[2],
            "scale_16": downsample_sizes[3],
        }

        if fp32_feature_prep:
            feature_device = padded_images[0].device
            with torch.autocast(device_type=feature_device.type, enabled=False):
                encoded_images = self._encode_images([image.float() for image in padded_images], target_sizes)
                reference_scales, support_scales = self._prepare_scale_features(
                    encoded_images,
                    (original_h, original_w),
                )
        else:
            encoded_images = self._encode_images(padded_images, target_sizes)
            reference_scales, support_scales = self._prepare_scale_features(
                encoded_images,
                (original_h, original_w),
            )
        reference_scales = _cast_feature_scales_to_float32(reference_scales)
        support_scales = _cast_nested_feature_scales_to_float32(support_scales)

        intermediate_preds: List[Dict[str, Tensor]] = []
        decoder_device = reference_scales[0].device
        with torch.autocast(device_type=decoder_device.type, enabled=False):
            current_field = torch.zeros(
                batch,
                2,
                *reference_scales[0].shape[-2:],
                device=decoder_device,
                dtype=torch.float32,
            )
            hidden = torch.zeros(
                batch,
                self.decoder_stages[0].gru.hidden_dim,
                *reference_scales[0].shape[-2:],
                device=decoder_device,
                dtype=torch.float32,
            )

            for scale_index, decoder_stage in enumerate(self.decoder_stages):
                reference_features = reference_scales[scale_index]
                scale_size = reference_features.shape[-2:]
                if scale_index > 0:
                    current_field = _resize_vector_field(current_field, scale_size)
                    hidden = F.interpolate(hidden, size=scale_size, mode="bilinear", align_corners=False)
                    hidden = self.hidden_adapters[scale_index - 1](hidden)

                for _ in range(iters_per_scale):
                    support_features = self._aggregate_support(
                        support_scales[scale_index],
                        current_field,
                        task_type=task_type,
                    )
                    hidden, current_field = decoder_stage(
                        reference=reference_features,
                        support=support_features,
                        hidden=hidden,
                        field=current_field,
                    )
                    intermediate_preds.append(
                        {
                            "pred_field": current_field,
                        }
                    )

            resolved_mode = _resolve_evidential_mode(use_niw=use_niw, evidential_mode=self.evidential_mode)
            if resolved_mode in {"niw", "mvder"}:
                evidence_head = self.final_mvder_head if resolved_mode == "mvder" else self.final_niw_head
                raw_niw = evidence_head(hidden, current_field)
                niw = unpack_niw_2d(
                    raw_niw,
                    r=self.niw_r,
                    eps=self.niw_eps,
                    nu_max=self.niw_nu_max,
                    nu_min=self.niw_nu_min,
                )
                if sef_grad_mode == "head_only":
                    raw_niw_sef = evidence_head(hidden.detach(), current_field.detach())
                    niw_sef = unpack_niw_2d(
                        raw_niw_sef,
                        r=self.niw_r,
                        eps=self.niw_eps,
                        nu_max=self.niw_nu_max,
                        nu_min=self.niw_nu_min,
                    )
                else:
                    raw_niw_sef = None
                    niw_sef = None
                final_pred = niw["mean"]
                aleatoric_cov = _covariance_to_legacy_layout(niw["C_ale"])
                epistemic_cov = _covariance_to_legacy_layout(niw["C_epi"])
                total_cov = _covariance_to_legacy_layout(niw["C_total"])
                aleatoric_unc = niw["U_ale"]
                epistemic_unc = niw["U_epi"]
                total_unc = niw["U_total"]
                risk_score_crp = self.risk_projector(niw) if self.risk_projector is not None else None
                raw_nig = None
                nig = None
            elif resolved_mode == "nig":
                raw_nig = self.final_nig_head(hidden, current_field)
                nig = unpack_nig_diag_2d(raw_nig, eps=self.niw_eps)
                final_pred = nig["mean"]
                aleatoric_cov = _diag_to_legacy_covariance(nig["diag_ale"])
                epistemic_cov = _diag_to_legacy_covariance(nig["diag_epi"])
                total_cov = _diag_to_legacy_covariance(nig["diag_total"])
                aleatoric_unc = nig["U_ale"]
                epistemic_unc = nig["U_epi"]
                total_unc = nig["U_total"]
                risk_score_crp = None
                raw_niw = None
                niw = None
                raw_niw_sef = None
                niw_sef = None
                raw_gaussian = None
                gaussian = None
            elif resolved_mode == "gaussian":
                raw_gaussian = self.final_gaussian_head(hidden, current_field)
                if self.gaussian_use_niw_mean:
                    raw_niw_for_mean = self.final_niw_head(hidden, current_field)
                    niw_for_mean = unpack_niw_2d(
                        raw_niw_for_mean,
                        r=self.niw_r,
                        eps=self.niw_eps,
                        nu_max=self.niw_nu_max,
                        nu_min=self.niw_nu_min,
                    )
                    raw_gaussian = raw_gaussian.clone()
                    raw_gaussian[:, 0:2] = niw_for_mean["mean"].detach()
                elif self.gaussian_use_base_mean:
                    raw_gaussian = raw_gaussian.clone()
                    raw_gaussian[:, 0:2] = current_field
                gaussian = unpack_gaussian_full_cov_2d(raw_gaussian, eps=self.niw_eps)
                final_pred = gaussian["mean"]
                aleatoric_cov = None
                epistemic_cov = None
                total_cov = _covariance_to_legacy_layout(gaussian["covariance"])
                aleatoric_unc = None
                epistemic_unc = None
                total_unc = gaussian["U_total"]
                risk_score_crp = None
                raw_niw = None
                niw = None
                raw_niw_sef = None
                niw_sef = None
                raw_nig = None
                nig = None
            else:
                raw_niw = None
                niw = None
                raw_nig = None
                nig = None
                raw_niw_sef = None
                niw_sef = None
                raw_gaussian = None
                gaussian = None
                final_pred = current_field
                aleatoric_cov = None
                epistemic_cov = None
                total_cov = None
                aleatoric_unc = None
                epistemic_unc = None
                total_unc = None
                risk_score_crp = None

        outputs = {
            "pred_field": final_pred,
            "evidential_mode": resolved_mode,
            "task_type": task_type,
            "intermediate_preds": intermediate_preds,
        }
        if resolved_mode in {"niw", "mvder"}:
            outputs.update(
                {
                    "raw_niw": raw_niw,
                    "niw_params": raw_niw,
                    "niw": niw,
                    "raw_niw_sef": raw_niw_sef,
                    "niw_sef": niw_sef,
                    "aleatoric_cov": aleatoric_cov,
                    "epistemic_cov": epistemic_cov,
                    "total_cov": total_cov,
                    "aleatoric_unc": aleatoric_unc,
                    "epistemic_unc": epistemic_unc,
                    "total_unc": total_unc,
                    "risk_score_crp": risk_score_crp,
                }
            )
        elif resolved_mode == "nig":
            outputs.update(
                {
                    "raw_nig": raw_nig,
                    "nig_params": raw_nig,
                    "nig": nig,
                    "aleatoric_cov": aleatoric_cov,
                    "epistemic_cov": epistemic_cov,
                    "total_cov": total_cov,
                    "aleatoric_unc": aleatoric_unc,
                    "epistemic_unc": epistemic_unc,
                    "total_unc": total_unc,
                }
            )
        elif resolved_mode == "gaussian":
            outputs.update(
                {
                    "raw_gaussian": raw_gaussian,
                    "gaussian_params": raw_gaussian,
                    "gaussian": gaussian,
                    "total_cov": total_cov,
                    "total_unc": total_unc,
                }
            )
        return outputs
