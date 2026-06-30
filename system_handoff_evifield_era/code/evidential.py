"""2D NIW evidential math utilities for dense vector fields."""

from __future__ import annotations

import math
from typing import Dict, Mapping, Optional, Tuple

import torch
from torch import Tensor
import torch.nn.functional as F


EPS = 1e-6
NU_MIN = 3.01


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _as_bhw_mask(valid_mask: Tensor | None, batch: int, height: int, width: int) -> Tensor | None:
    if valid_mask is None:
        return None
    if valid_mask.ndim == 4:
        _require(
            valid_mask.shape == (batch, 1, height, width),
            f"Expected valid_mask [B, 1, H, W], got {tuple(valid_mask.shape)}.",
        )
        return valid_mask[:, 0]
    _require(
        valid_mask.shape == (batch, height, width),
        f"Expected valid_mask [B, H, W], got {tuple(valid_mask.shape)}.",
    )
    return valid_mask


def _safe_reduce(loss_map: Tensor, valid_mask: Tensor | None, reduction: str) -> Tensor:
    if reduction == "none":
        return loss_map
    if valid_mask is None:
        if reduction == "mean":
            return loss_map.mean()
        if reduction == "sum":
            return loss_map.sum()
        raise ValueError(f"Unsupported reduction='{reduction}'.")
    weights = valid_mask.unsqueeze(1).expand_as(loss_map)
    weighted = loss_map * weights
    if reduction == "mean":
        return weighted.sum() / weights.sum().clamp_min(1.0)
    if reduction == "sum":
        return weighted.sum()
    raise ValueError(f"Unsupported reduction='{reduction}'.")


def _cov_trace_to_uncertainty(covariance: Tensor) -> Tensor:
    return (0.5 * (covariance[..., 0, 0] + covariance[..., 1, 1])).unsqueeze(1)


def niw_covariance_risk_features(
    niw_outputs: Mapping[str, Tensor],
    eps: float = EPS,
    feature_set: str = "covariance",
) -> Tensor:
    """Build the covariance feature stack used by the CRP head."""
    normalized = feature_set.lower().replace("-", "_")
    _require(
        normalized
        in {
            "covariance",
            "full",
            "no_coupling",
            "strict_no_coupling",
            "evidence_only",
            "evidence_trace",
            "monotonic_evidence",
        },
        f"Unsupported NIW covariance risk feature_set='{feature_set}'.",
    )
    required = ("C_total", "C_ale", "C_epi", "kappa", "nu")
    for key in required:
        _require(isinstance(niw_outputs.get(key), Tensor), f"NIW outputs must contain tensor '{key}'.")

    cov_total = torch.nan_to_num(niw_outputs["C_total"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    cov_ale = torch.nan_to_num(niw_outputs["C_ale"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    cov_epi = torch.nan_to_num(niw_outputs["C_epi"].float(), nan=0.0, posinf=0.0, neginf=0.0)
    _require(cov_total.ndim == 5 and cov_total.shape[-2:] == (2, 2), f"Expected C_total [B, H, W, 2, 2], got {tuple(cov_total.shape)}.")
    _require(cov_ale.shape == cov_total.shape, "C_ale and C_total shape mismatch.")
    _require(cov_epi.shape == cov_total.shape, "C_epi and C_total shape mismatch.")

    c00 = cov_total[..., 0, 0]
    c01 = cov_total[..., 0, 1]
    c11 = cov_total[..., 1, 1]
    trace_total = (c00 + c11).clamp_min(0.0)
    trace_ale = (cov_ale[..., 0, 0] + cov_ale[..., 1, 1]).clamp_min(0.0)
    trace_epi = (cov_epi[..., 0, 0] + cov_epi[..., 1, 1]).clamp_min(0.0)
    disc = ((c00 - c11).square() + 4.0 * c01.square()).clamp_min(0.0).sqrt()
    full_lambda_max = (0.5 * (trace_total + disc)).clamp_min(eps)
    full_lambda_min = (0.5 * (trace_total - disc)).clamp_min(eps)
    if normalized == "strict_no_coupling":
        lambda_max = torch.maximum(c00, c11).clamp_min(eps)
        lambda_min = torch.minimum(c00, c11).clamp_min(eps)
    else:
        lambda_max = full_lambda_max
        lambda_min = full_lambda_min

    kappa = torch.nan_to_num(niw_outputs["kappa"].float(), nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    nu = torch.nan_to_num(niw_outputs["nu"].float(), nan=3.0 + eps, posinf=3.0 + eps, neginf=3.0 + eps)
    nu_minus_d_minus_one = (nu - 3.0 + eps).clamp_min(eps)

    evidence_maps = [
        torch.log1p(kappa),
        torch.log1p(nu_minus_d_minus_one),
    ]
    if normalized == "evidence_only":
        feature_maps = evidence_maps
    elif normalized in {"evidence_trace", "monotonic_evidence"}:
        feature_maps = [
            torch.log1p(trace_total).unsqueeze(1),
            *evidence_maps,
        ]
        if normalized == "monotonic_evidence":
            feature_maps = evidence_maps
    else:
        feature_maps = [
            torch.log1p(trace_total).unsqueeze(1),
            torch.log1p(trace_ale).unsqueeze(1),
            torch.log1p(trace_epi).unsqueeze(1),
            torch.log1p(lambda_max).unsqueeze(1),
            torch.log1p(lambda_min).unsqueeze(1),
        ]
    if normalized in {"covariance", "full"}:
        anisotropy = (full_lambda_max / full_lambda_min.clamp_min(eps)).clamp_min(0.0)
        corr_uv = c01 / (c00.clamp_min(eps).sqrt() * c11.clamp_min(eps).sqrt()).clamp_min(eps)
        feature_maps.extend(
            [
                torch.log1p(anisotropy).unsqueeze(1),
                torch.log1p(c01.abs()).unsqueeze(1),
                corr_uv.clamp(-1.0, 1.0).abs().unsqueeze(1),
            ]
        )
    if normalized in {"covariance", "full", "no_coupling", "strict_no_coupling"}:
        feature_maps.extend(evidence_maps)
    features = torch.cat(feature_maps, dim=1)
    return torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)


def niw_risk_score_map(outputs: Mapping[str, Tensor | Mapping[str, Tensor]], mode: str, eps: float = EPS) -> Tensor:
    """Select a scalar risk score map from NIW model outputs."""
    normalized = mode.lower().replace("-", "_")
    if normalized in {"trace", "trace_total", "total_trace", "native"}:
        niw = outputs.get("niw")
        _require(isinstance(niw, Mapping), "trace risk score requires outputs['niw'].")
        cov_total = niw.get("C_total")
        _require(isinstance(cov_total, Tensor), "trace risk score requires outputs['niw']['C_total'].")
        safe_cov = torch.nan_to_num(cov_total.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return (safe_cov[..., 0, 0] + safe_cov[..., 1, 1]).clamp_min(0.0).unsqueeze(1)
    if normalized in {"abs_uv", "abs_c_uv", "abs_cov_uv"}:
        niw = outputs.get("niw")
        _require(isinstance(niw, Mapping), "abs_uv risk score requires outputs['niw'].")
        cov_total = niw.get("C_total")
        _require(isinstance(cov_total, Tensor), "abs_uv risk score requires outputs['niw']['C_total'].")
        safe_cov = torch.nan_to_num(cov_total.float(), nan=0.0, posinf=0.0, neginf=0.0)
        return safe_cov[..., 0, 1].abs().unsqueeze(1)
    if normalized in {"analytic_evidence", "evidence", "evidence_strength"}:
        niw = outputs.get("niw")
        _require(isinstance(niw, Mapping), "analytic_evidence risk score requires outputs['niw'].")
        kappa = niw.get("kappa")
        nu = niw.get("nu")
        _require(isinstance(kappa, Tensor), "analytic_evidence risk score requires outputs['niw']['kappa'].")
        _require(isinstance(nu, Tensor), "analytic_evidence risk score requires outputs['niw']['nu'].")
        safe_kappa = torch.nan_to_num(kappa.float(), nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
        safe_nu = torch.nan_to_num(nu.float(), nan=3.0 + eps, posinf=3.0 + eps, neginf=3.0 + eps)
        nu_minus_d_minus_one = (safe_nu - 3.0 + eps).clamp_min(eps)
        return -torch.log1p(safe_kappa) - torch.log1p(nu_minus_d_minus_one)
    if normalized in {"crp", "risk_score_crp", "evidence_trace", "monotonic_evidence"}:
        score = outputs.get("risk_score_crp")
        _require(isinstance(score, Tensor), "CRP risk score requires outputs['risk_score_crp'].")
        return torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    raise ValueError(f"Unsupported risk_score_mode='{mode}'. Expected trace, abs_uv, analytic_evidence, or crp.")


def nig_risk_score_map(outputs: Mapping[str, Tensor | Mapping[str, Tensor]], mode: str, eps: float = EPS) -> Tensor:
    """Select a scalar risk score map from component-wise NIG model outputs."""
    normalized = mode.lower().replace("-", "_")
    if normalized in {"nig_original", "nig_uncertainty", "component_uncertainty", "trace", "native"}:
        score = outputs.get("total_unc")
        _require(isinstance(score, Tensor), "NIG original uncertainty requires outputs['total_unc'].")
        return torch.nan_to_num(score.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    if normalized in {"nig_e", "nig_evidence", "nig_evidence_strength"}:
        nig = outputs.get("nig")
        _require(isinstance(nig, Mapping), "NIG evidence score requires outputs['nig'].")
        v = nig.get("v")
        alpha = nig.get("alpha")
        _require(isinstance(v, Tensor), "NIG evidence score requires outputs['nig']['v'].")
        _require(isinstance(alpha, Tensor), "NIG evidence score requires outputs['nig']['alpha'].")
        safe_v = torch.nan_to_num(v.float(), nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
        safe_alpha = torch.nan_to_num(alpha.float(), nan=1.0 + eps, posinf=1.0 + eps, neginf=1.0 + eps).clamp_min(1.0 + eps)
        return -0.5 * torch.log1p(safe_v).sum(dim=1, keepdim=True) - 0.5 * torch.log1p(safe_alpha).sum(dim=1, keepdim=True)
    raise ValueError(f"Unsupported NIG risk_score_mode='{mode}'. Expected nig_original or nig_evidence.")


def gaussian_risk_score_map(outputs: Mapping[str, Tensor | Mapping[str, Tensor]], mode: str, eps: float = EPS) -> Tensor:
    """Select a scalar risk score map from full-covariance Gaussian outputs."""
    normalized = mode.lower().replace("-", "_")
    gaussian = outputs.get("gaussian")
    _require(isinstance(gaussian, Mapping), "Gaussian risk score requires outputs['gaussian'].")
    covariance = gaussian.get("covariance")
    _require(isinstance(covariance, Tensor), "Gaussian risk score requires outputs['gaussian']['covariance'].")
    safe_cov = torch.nan_to_num(covariance.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if normalized in {"trace", "trace_total", "total_trace", "native"}:
        return (safe_cov[..., 0, 0] + safe_cov[..., 1, 1]).clamp_min(0.0).unsqueeze(1)
    if normalized in {"abs_uv", "abs_c_uv", "abs_cov_uv"}:
        return safe_cov[..., 0, 1].abs().unsqueeze(1)
    raise ValueError(f"Unsupported Gaussian risk_score_mode='{mode}'. Expected trace or abs_uv.")


def unpack_gaussian_full_cov_2d(raw: Tensor, eps: float = EPS) -> Dict[str, Tensor]:
    """Convert raw full-covariance Gaussian outputs into mean and covariance."""
    _require(
        raw.ndim == 4 and raw.shape[1] == 5,
        f"Expected raw Gaussian tensor [B, 5, H, W], got {tuple(raw.shape)}.",
    )
    safe_raw = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = safe_raw[:, 0:2]
    l11 = F.softplus(safe_raw[:, 2:3]) + eps
    l21 = safe_raw[:, 3:4]
    l22 = F.softplus(safe_raw[:, 4:5]) + eps

    l11_hw = l11[:, 0]
    l21_hw = l21[:, 0]
    l22_hw = l22[:, 0]
    zeros = torch.zeros_like(l11_hw)
    row0 = torch.stack([l11_hw, zeros], dim=-1)
    row1 = torch.stack([l21_hw, l22_hw], dim=-1)
    chol = torch.stack([row0, row1], dim=-2)
    covariance = torch.matmul(chol, chol.transpose(-1, -2))
    eye = torch.eye(2, device=covariance.device, dtype=covariance.dtype).view(1, 1, 1, 2, 2)
    covariance = covariance + eps * eye
    uncertainty = _cov_trace_to_uncertainty(covariance)
    return {
        "mean": torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0),
        "L": torch.nan_to_num(chol, nan=0.0, posinf=0.0, neginf=0.0),
        "covariance": torch.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0),
        "U_total": torch.nan_to_num(uncertainty, nan=0.0, posinf=0.0, neginf=0.0),
    }


def gaussian_nll_2d(
    mean: Tensor,
    covariance: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    eps: float = EPS,
) -> Tuple[Tensor, Tensor]:
    """Full-covariance 2D Gaussian negative log-likelihood."""
    _require(mean.shape == target.shape, f"Expected mean/target shape match, got {tuple(mean.shape)} vs {tuple(target.shape)}.")
    _require(mean.ndim == 4 and mean.shape[1] == 2, f"Expected mean [B, 2, H, W], got {tuple(mean.shape)}.")
    _require(
        covariance.shape == (mean.shape[0], mean.shape[2], mean.shape[3], 2, 2),
        f"Expected covariance [B, H, W, 2, 2], got {tuple(covariance.shape)}.",
    )
    batch, _, height, width = mean.shape
    safe_mean = torch.nan_to_num(mean.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_cov = torch.nan_to_num(covariance.float(), nan=0.0, posinf=0.0, neginf=0.0)
    eye = torch.eye(2, device=safe_cov.device, dtype=safe_cov.dtype).view(1, 1, 1, 2, 2)
    safe_cov = safe_cov + eps * eye
    residual = (safe_target - safe_mean).permute(0, 2, 3, 1).unsqueeze(-1)
    solved = torch.linalg.solve(safe_cov, residual)
    quadratic = torch.matmul(residual.transpose(-1, -2), solved).squeeze(-1).squeeze(-1).unsqueeze(1)
    sign, logabsdet = torch.linalg.slogdet(safe_cov)
    safe_logdet = torch.where(sign > 0.0, logabsdet, torch.full_like(logabsdet, math.log(eps))).unsqueeze(1)
    nll_map = 0.5 * (quadratic + safe_logdet + 2.0 * math.log(2.0 * math.pi))
    nll_map = torch.nan_to_num(nll_map, nan=0.0, posinf=0.0, neginf=0.0)
    mask_bhw = _as_bhw_mask(valid_mask.float() if valid_mask is not None else None, batch, height, width)
    nll_mean = _safe_reduce(nll_map, mask_bhw, reduction="mean")
    return nll_map, nll_mean


def unpack_niw_2d(
    raw: Tensor,
    r: float = 1.0,
    eps: float = EPS,
    nu_max: Optional[float] = None,
    nu_min: float = NU_MIN,
) -> Dict[str, Tensor]:
    """
    Convert raw NIW outputs into structured 2D statistics.

    raw: [B, 6, H, W] for the tied-nu dense NIW head, or [B, 7, H, W]
    for direct multivariate DER/NIW adaptation with independently predicted nu.
    """
    _require(
        raw.ndim == 4 and raw.shape[1] in {6, 7},
        f"Expected raw NIW tensor [B, 6 or 7, H, W], got {tuple(raw.shape)}.",
    )
    _require(r > 0.0, f"niw_r must be positive, got {r}.")
    _require(nu_min > 3.0, f"niw_nu_min must be greater than 3.0, got {nu_min}.")

    safe_raw = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = safe_raw[:, 0:2]
    kappa_raw = safe_raw[:, 2:3]
    if safe_raw.shape[1] == 7:
        nu_raw = safe_raw[:, 3:4]
        l11_raw = safe_raw[:, 4:5]
        l21_raw = safe_raw[:, 5:6]
        l22_raw = safe_raw[:, 6:7]
        direct_independent_nu = True
    else:
        nu_raw = None
        l11_raw = safe_raw[:, 3:4]
        l21_raw = safe_raw[:, 4:5]
        l22_raw = safe_raw[:, 5:6]
        direct_independent_nu = False

    min_nu = float(nu_min)
    if direct_independent_nu:
        kappa = F.softplus(kappa_raw) + eps
        if nu_max is not None:
            _require(nu_max > min_nu, f"niw_nu_max must be greater than {min_nu}, got {nu_max}.")
            nu = min_nu + (float(nu_max) - min_nu) * torch.sigmoid(nu_raw)
        else:
            nu = min_nu + F.softplus(nu_raw) + eps
    elif nu_max is not None:
        _require(nu_max > min_nu, f"niw_nu_max must be greater than {min_nu}, got {nu_max}.")
        nu = min_nu + (float(nu_max) - min_nu) * torch.sigmoid(kappa_raw)
        kappa = (nu / float(r)).clamp_min(eps)
    else:
        min_kappa = min_nu / float(r)
        kappa = min_kappa + F.softplus(kappa_raw) + eps
        nu = kappa * float(r)

    l11 = F.softplus(l11_raw) + eps
    l21 = l21_raw
    l22 = F.softplus(l22_raw) + eps

    l11_hw = l11[:, 0]
    l21_hw = l21[:, 0]
    l22_hw = l22[:, 0]
    zeros = torch.zeros_like(l11_hw)
    row0 = torch.stack([l11_hw, zeros], dim=-1)
    row1 = torch.stack([l21_hw, l22_hw], dim=-1)
    L = torch.stack([row0, row1], dim=-2)

    base_psi = torch.matmul(L, L.transpose(-1, -2))
    if direct_independent_nu:
        eye = torch.eye(2, device=base_psi.device, dtype=base_psi.dtype).view(1, 1, 1, 2, 2)
        psi = base_psi + eps * eye
    else:
        nu_hw = nu[:, 0]
        psi = nu_hw.unsqueeze(-1).unsqueeze(-1) * base_psi

    nu_minus_three = (nu - 3.0).clamp_min(eps)
    nu_minus_three_hw = nu_minus_three[:, 0].unsqueeze(-1).unsqueeze(-1)
    aleatoric_cov = psi / nu_minus_three_hw

    epi_denom = (kappa * nu_minus_three).clamp_min(eps)
    epistemic_cov = psi / epi_denom[:, 0].unsqueeze(-1).unsqueeze(-1)
    total_cov = aleatoric_cov + epistemic_cov

    aleatoric_unc = _cov_trace_to_uncertainty(aleatoric_cov)
    epistemic_unc = _cov_trace_to_uncertainty(epistemic_cov)
    total_unc = _cov_trace_to_uncertainty(total_cov)

    return {
        "mean": mean,
        "kappa": torch.nan_to_num(kappa, nan=eps, posinf=eps, neginf=eps),
        "nu": torch.nan_to_num(nu, nan=min_nu, posinf=min_nu, neginf=min_nu),
        "L": torch.nan_to_num(L, nan=0.0, posinf=0.0, neginf=0.0),
        "Psi": torch.nan_to_num(psi, nan=0.0, posinf=0.0, neginf=0.0),
        "C_ale": torch.nan_to_num(aleatoric_cov, nan=0.0, posinf=0.0, neginf=0.0),
        "C_epi": torch.nan_to_num(epistemic_cov, nan=0.0, posinf=0.0, neginf=0.0),
        "C_total": torch.nan_to_num(total_cov, nan=0.0, posinf=0.0, neginf=0.0),
        "U_ale": torch.nan_to_num(aleatoric_unc, nan=0.0, posinf=0.0, neginf=0.0),
        "U_epi": torch.nan_to_num(epistemic_unc, nan=0.0, posinf=0.0, neginf=0.0),
        "U_total": torch.nan_to_num(total_unc, nan=0.0, posinf=0.0, neginf=0.0),
    }


def niw_student_t_nll_2d(
    mean: Tensor,
    kappa: Tensor,
    nu: Tensor,
    Psi: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    eps: float = EPS,
) -> Tuple[Tensor, Tensor]:
    """
    2D Student-t NLL induced by a Normal-Inverse-Wishart posterior.

    mean: [B, 2, H, W]
    target: [B, 2, H, W]
    kappa, nu: [B, 1, H, W]
    Psi: [B, H, W, 2, 2]
    """
    _require(mean.shape == target.shape, f"Expected mean/target shape match, got {tuple(mean.shape)} vs {tuple(target.shape)}.")
    _require(mean.ndim == 4 and mean.shape[1] == 2, f"Expected mean [B, 2, H, W], got {tuple(mean.shape)}.")
    _require(kappa.shape == nu.shape == (mean.shape[0], 1, mean.shape[2], mean.shape[3]), "kappa/nu shape mismatch.")
    _require(
        Psi.shape == (mean.shape[0], mean.shape[2], mean.shape[3], 2, 2),
        f"Expected Psi [B, H, W, 2, 2], got {tuple(Psi.shape)}.",
    )

    batch, _, height, width = mean.shape
    safe_mean = torch.nan_to_num(mean.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_kappa = torch.nan_to_num(kappa.float(), nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    safe_nu = torch.nan_to_num(nu.float(), nan=NU_MIN, posinf=NU_MIN, neginf=NU_MIN).clamp_min(NU_MIN)
    safe_psi = torch.nan_to_num(Psi.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mask_bhw = _as_bhw_mask(valid_mask.float() if valid_mask is not None else None, batch, height, width)

    dof = (safe_nu - 1.0).clamp_min(2.0 + eps)
    scale = ((safe_kappa + 1.0) / (safe_kappa * dof)).clamp_min(eps)

    scale_matrix = scale[:, 0].unsqueeze(-1).unsqueeze(-1) * safe_psi
    eye = torch.eye(2, device=scale_matrix.device, dtype=scale_matrix.dtype).view(1, 1, 1, 2, 2)
    scale_matrix = scale_matrix + eps * eye

    residual = (safe_target - safe_mean).permute(0, 2, 3, 1).unsqueeze(-1)
    solved = torch.linalg.solve(scale_matrix, residual)
    quadratic = torch.matmul(residual.transpose(-1, -2), solved).squeeze(-1).squeeze(-1).unsqueeze(1)

    sign, logabsdet = torch.linalg.slogdet(scale_matrix)
    safe_logdet = torch.where(sign > 0.0, logabsdet, torch.full_like(logabsdet, math.log(eps))).unsqueeze(1)

    nll_map = (
        -torch.lgamma((safe_nu + 1.0) / 2.0)
        + torch.lgamma((safe_nu - 1.0) / 2.0)
        + torch.log((safe_nu - 1.0).clamp_min(eps) * math.pi)
        + 0.5 * safe_logdet
        + ((safe_nu + 1.0) / 2.0) * torch.log1p(quadratic / (safe_nu - 1.0).clamp_min(eps))
    )
    nll_map = torch.nan_to_num(nll_map, nan=0.0, posinf=0.0, neginf=0.0)
    nll_mean = _safe_reduce(nll_map, mask_bhw, reduction="mean")
    return nll_map, nll_mean


def unpack_nig_diag_2d(
    raw: Tensor,
    eps: float = EPS,
) -> Dict[str, Tensor]:
    """
    Convert raw 8-channel component-wise NIG outputs into diagonal statistics.

    raw: [B, 8, H, W] with [mean_u, mean_v, v_u, v_v, alpha_u, alpha_v, beta_u, beta_v].
    """
    _require(raw.ndim == 4 and raw.shape[1] == 8, f"Expected raw NIG tensor [B, 8, H, W], got {tuple(raw.shape)}.")

    safe_raw = torch.nan_to_num(raw.float(), nan=0.0, posinf=0.0, neginf=0.0)
    mean = safe_raw[:, 0:2]
    v = F.softplus(safe_raw[:, 2:4]) + eps
    alpha = 1.0 + F.softplus(safe_raw[:, 4:6]) + eps
    beta = F.softplus(safe_raw[:, 6:8]) + eps

    alpha_minus_one = (alpha - 1.0).clamp_min(eps)
    aleatoric = beta / alpha_minus_one
    epistemic = beta / (v * alpha_minus_one).clamp_min(eps)
    total = aleatoric + epistemic

    return {
        "mean": mean,
        "v": torch.nan_to_num(v, nan=eps, posinf=eps, neginf=eps),
        "alpha": torch.nan_to_num(alpha, nan=1.0 + eps, posinf=1.0 + eps, neginf=1.0 + eps),
        "beta": torch.nan_to_num(beta, nan=eps, posinf=eps, neginf=eps),
        "diag_ale": torch.nan_to_num(aleatoric, nan=0.0, posinf=0.0, neginf=0.0),
        "diag_epi": torch.nan_to_num(epistemic, nan=0.0, posinf=0.0, neginf=0.0),
        "diag_total": torch.nan_to_num(total, nan=0.0, posinf=0.0, neginf=0.0),
        "U_ale": torch.nan_to_num(aleatoric.mean(dim=1, keepdim=True), nan=0.0, posinf=0.0, neginf=0.0),
        "U_epi": torch.nan_to_num(epistemic.mean(dim=1, keepdim=True), nan=0.0, posinf=0.0, neginf=0.0),
        "U_total": torch.nan_to_num(total.mean(dim=1, keepdim=True), nan=0.0, posinf=0.0, neginf=0.0),
    }


def nig_student_t_nll_diag_2d(
    mean: Tensor,
    v: Tensor,
    alpha: Tensor,
    beta: Tensor,
    target: Tensor,
    valid_mask: Tensor | None = None,
    eps: float = EPS,
) -> Tuple[Tensor, Tensor]:
    """Factorized 2D Student-t NLL induced by independent per-component NIG posteriors."""
    _require(mean.shape == target.shape, f"Expected mean/target shape match, got {tuple(mean.shape)} vs {tuple(target.shape)}.")
    _require(mean.ndim == 4 and mean.shape[1] == 2, f"Expected mean [B, 2, H, W], got {tuple(mean.shape)}.")
    _require(v.shape == alpha.shape == beta.shape == mean.shape, "NIG v/alpha/beta shape mismatch.")

    batch, _, height, width = mean.shape
    safe_mean = torch.nan_to_num(mean.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_v = torch.nan_to_num(v.float(), nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    safe_alpha = torch.nan_to_num(alpha.float(), nan=1.0 + eps, posinf=1.0 + eps, neginf=1.0 + eps).clamp_min(1.0 + eps)
    safe_beta = torch.nan_to_num(beta.float(), nan=eps, posinf=eps, neginf=eps).clamp_min(eps)
    mask_bhw = _as_bhw_mask(valid_mask.float() if valid_mask is not None else None, batch, height, width)

    dof = (2.0 * safe_alpha).clamp_min(2.0 + eps)
    scale = torch.sqrt((safe_beta * (safe_v + 1.0) / (safe_v * safe_alpha).clamp_min(eps)).clamp_min(eps))
    residual = safe_target - safe_mean
    log_prob = (
        torch.lgamma((dof + 1.0) / 2.0)
        - torch.lgamma(dof / 2.0)
        - 0.5 * (torch.log(dof * math.pi) + 2.0 * torch.log(scale.clamp_min(eps)))
        - ((dof + 1.0) / 2.0) * torch.log1p((residual / scale.clamp_min(eps)).square() / dof)
    )
    nll_map = torch.nan_to_num(-log_prob.sum(dim=1, keepdim=True), nan=0.0, posinf=0.0, neginf=0.0)
    nll_mean = _safe_reduce(nll_map, mask_bhw, reduction="mean")
    return nll_map, nll_mean


def structured_uncertainty_smoothness(
    U_total: Tensor,
    pred_field: Tensor,
    ref_image: Tensor | None = None,
    valid_mask: Tensor | None = None,
    sigma_image: float = 0.1,
    sigma_vector: float = 1.0,
    eps: float = EPS,
    detach_weights: bool = True,
    return_stats: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    """
    Edge-aware SEF-light regularizer on log total uncertainty.

    U_total: [B, 1, H, W]
    pred_field: [B, 2, H, W]
    ref_image: [B, C, H, W] or None
    valid_mask: [B, 1, H, W] or [B, H, W]
    """
    _require(U_total.ndim == 4 and U_total.shape[1] == 1, f"Expected U_total [B, 1, H, W], got {tuple(U_total.shape)}.")
    _require(
        pred_field.ndim == 4 and pred_field.shape[1] == 2 and pred_field.shape[0] == U_total.shape[0] and pred_field.shape[-2:] == U_total.shape[-2:],
        f"Expected pred_field [B, 2, H, W] aligned with U_total, got {tuple(pred_field.shape)} vs {tuple(U_total.shape)}.",
    )
    if ref_image is not None:
        _require(
            ref_image.ndim == 4 and ref_image.shape[0] == U_total.shape[0] and ref_image.shape[-2:] == U_total.shape[-2:],
            f"Expected ref_image [B, C, H, W] aligned with U_total, got {tuple(ref_image.shape)}.",
        )
    _require(sigma_image > 0.0, f"sigma_image must be positive, got {sigma_image}.")
    _require(sigma_vector > 0.0, f"sigma_vector must be positive, got {sigma_vector}.")

    safe_uncertainty = torch.nan_to_num(U_total.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
    safe_pred = torch.nan_to_num(pred_field.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_ref = torch.nan_to_num(ref_image.float(), nan=0.0, posinf=0.0, neginf=0.0) if ref_image is not None else None
    batch, _, height, width = safe_uncertainty.shape
    mask_bhw = _as_bhw_mask(valid_mask.float() if valid_mask is not None else None, batch, height, width)

    if detach_weights:
        safe_pred = safe_pred.detach()
        if safe_ref is not None:
            safe_ref = safe_ref.detach()
        if mask_bhw is not None:
            mask_bhw = mask_bhw.detach()

    log_uncertainty = torch.log(safe_uncertainty)
    valid_edge_count = safe_uncertainty.new_zeros(())
    total_loss = safe_uncertainty.new_zeros(())

    def _accumulate(horizontal: bool) -> None:
        nonlocal valid_edge_count, total_loss
        if horizontal:
            unc_a = log_uncertainty[:, :, :, :-1]
            unc_b = log_uncertainty[:, :, :, 1:]
            vec_a = safe_pred[:, :, :, :-1]
            vec_b = safe_pred[:, :, :, 1:]
            if safe_ref is not None:
                img_a = safe_ref[:, :, :, :-1]
                img_b = safe_ref[:, :, :, 1:]
        else:
            unc_a = log_uncertainty[:, :, :-1, :]
            unc_b = log_uncertainty[:, :, 1:, :]
            vec_a = safe_pred[:, :, :-1, :]
            vec_b = safe_pred[:, :, 1:, :]
            if safe_ref is not None:
                img_a = safe_ref[:, :, :-1, :]
                img_b = safe_ref[:, :, 1:, :]

        vec_term = (vec_a - vec_b).square().sum(dim=1, keepdim=True) / (sigma_vector**2)
        image_term = 0.0
        if safe_ref is not None:
            image_term = (img_a - img_b).square().mean(dim=1, keepdim=True) / (sigma_image**2)
        pair_weight = torch.exp(-(vec_term + image_term))
        if mask_bhw is not None:
            if horizontal:
                pair_valid = (mask_bhw[:, :, :-1] > 0.5) & (mask_bhw[:, :, 1:] > 0.5)
            else:
                pair_valid = (mask_bhw[:, :-1, :] > 0.5) & (mask_bhw[:, 1:, :] > 0.5)
            pair_valid_weight = pair_valid.unsqueeze(1).float()
        else:
            pair_valid_weight = torch.ones_like(pair_weight)
        pair_weight = pair_weight * pair_valid_weight
        if detach_weights:
            pair_weight = pair_weight.detach()

        diff = unc_a - unc_b
        penalty = torch.sqrt(diff.square() + eps**2)
        total_loss = total_loss + (pair_weight * penalty).sum()
        valid_edge_count = valid_edge_count + pair_valid_weight.sum()

    _accumulate(horizontal=True)
    _accumulate(horizontal=False)
    if valid_edge_count <= 0:
        loss = U_total.sum() * 0.0
    else:
        loss = total_loss / valid_edge_count.clamp_min(1.0)
    if not return_stats:
        return loss
    stats = {
        "valid_edges": valid_edge_count.detach(),
        "u_total_mean": safe_uncertainty.mean().detach(),
        "u_total_max": safe_uncertainty.amax().detach(),
    }
    return loss, stats


def _replicate_avg_pool2d(value: Tensor, kernel_size: int) -> Tensor:
    _require(kernel_size > 0 and kernel_size % 2 == 1, f"kernel_size must be a positive odd integer, got {kernel_size}.")
    pad = kernel_size // 2
    padded = F.pad(value, (pad, pad, pad, pad), mode="replicate")
    return F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)


def _forward_gradient_magnitude(value: Tensor, eps: float) -> Tensor:
    dx = torch.zeros_like(value)
    dy = torch.zeros_like(value)
    dx[..., :, :-1] = value[..., :, 1:] - value[..., :, :-1]
    dy[..., :-1, :] = value[..., 1:, :] - value[..., :-1, :]
    return torch.sqrt(dx.square().mean(dim=1, keepdim=True) + dy.square().mean(dim=1, keepdim=True) + eps**2)


def _normalize_boundary_map(boundary: Tensor, valid_mask: Tensor | None, eps: float) -> Tensor:
    if valid_mask is None:
        denom = boundary.flatten(1).mean(dim=1).view(-1, 1, 1, 1).clamp_min(eps)
        return boundary / denom
    weights = valid_mask.float()
    denom = (boundary * weights).flatten(1).sum(dim=1) / weights.flatten(1).sum(dim=1).clamp_min(1.0)
    return boundary / denom.view(-1, 1, 1, 1).clamp_min(eps)


def isolated_evidence_spike_suppression(
    U_total: Tensor,
    pred_field: Tensor,
    target: Tensor,
    ref_image: Tensor | None = None,
    valid_mask: Tensor | None = None,
    delta: float = 1.0,
    sigma_error: float = 1.0,
    sigma_boundary: float = 1.0,
    eps: float = EPS,
    kernel_size: int = 3,
    return_stats: bool = False,
) -> Tensor | Tuple[Tensor, Dict[str, Tensor]]:
    """
    SEF-v2 isolated high-evidence-spike regularizer.

    The gates are detached by construction. This keeps the term MAP-prior-like and
    prevents the model from reducing the penalty by changing the gate inputs.
    """
    _require(U_total.ndim == 4 and U_total.shape[1] == 1, f"Expected U_total [B, 1, H, W], got {tuple(U_total.shape)}.")
    _require(
        pred_field.ndim == 4 and pred_field.shape[1] == 2 and pred_field.shape == target.shape,
        f"Expected pred_field/target [B, 2, H, W], got {tuple(pred_field.shape)} vs {tuple(target.shape)}.",
    )
    _require(
        U_total.shape[0] == pred_field.shape[0] and U_total.shape[-2:] == pred_field.shape[-2:],
        f"U_total and pred_field must align spatially, got {tuple(U_total.shape)} vs {tuple(pred_field.shape)}.",
    )
    if ref_image is not None:
        _require(
            ref_image.ndim == 4 and ref_image.shape[0] == U_total.shape[0] and ref_image.shape[-2:] == U_total.shape[-2:],
            f"Expected ref_image [B, C, H, W] aligned with U_total, got {tuple(ref_image.shape)}.",
        )
    _require(delta >= 0.0, f"delta must be non-negative, got {delta}.")
    _require(sigma_error > 0.0, f"sigma_error must be positive, got {sigma_error}.")
    _require(sigma_boundary > 0.0, f"sigma_boundary must be positive, got {sigma_boundary}.")
    _require(eps > 0.0, f"eps must be positive, got {eps}.")

    safe_uncertainty = torch.nan_to_num(U_total.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(eps)
    safe_pred = torch.nan_to_num(pred_field.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_target = torch.nan_to_num(target.float(), nan=0.0, posinf=0.0, neginf=0.0)
    safe_ref = torch.nan_to_num(ref_image.float(), nan=0.0, posinf=0.0, neginf=0.0) if ref_image is not None else None
    batch, _, height, width = safe_uncertainty.shape
    mask_bhw = _as_bhw_mask(valid_mask.float() if valid_mask is not None else None, batch, height, width)
    valid_weight = mask_bhw.unsqueeze(1).float() if mask_bhw is not None else torch.ones_like(safe_uncertainty)
    valid_weight = valid_weight.detach()

    z = torch.log(safe_uncertainty + eps)
    z_ref = _replicate_avg_pool2d(z.detach(), kernel_size=kernel_size)
    spike = F.relu(z - z_ref - float(delta))

    error = torch.linalg.norm(safe_target - safe_pred, dim=1, keepdim=True).detach()
    pred_gradient = _forward_gradient_magnitude(safe_pred.detach(), eps=eps)
    if safe_ref is None:
        image_gradient = torch.zeros_like(pred_gradient)
    else:
        image_gradient = _forward_gradient_magnitude(safe_ref.detach(), eps=eps)
    boundary = _normalize_boundary_map(image_gradient + pred_gradient, valid_weight, eps=eps).detach()

    error_gate = torch.exp(-error / float(sigma_error))
    boundary_gate = torch.exp(-boundary / float(sigma_boundary))
    gate = (error_gate * boundary_gate).detach()

    weighted = valid_weight * gate * spike.square()
    raw_spike = valid_weight * spike.square()
    valid_count = valid_weight.sum()
    if not bool((valid_count > 0).detach().cpu().item()):
        loss = U_total.sum() * 0.0
    else:
        loss = weighted.sum() / valid_count.clamp_min(1.0)
    if not return_stats:
        return loss
    stats = {
        "valid_pixels": valid_count.detach(),
        "gate_mean": ((gate * valid_weight).sum() / valid_count.clamp_min(1.0)).detach(),
        "gate_p50": gate.new_zeros(()).detach(),
        "gate_p95": gate.new_zeros(()).detach(),
        "gate_p99": gate.new_zeros(()).detach(),
        "spike_frac": (((spike > 0.0).float() * valid_weight).sum() / valid_count.clamp_min(1.0)).detach(),
        "spike_mean": ((spike * valid_weight).sum() / valid_count.clamp_min(1.0)).detach(),
        "spike_max": spike.masked_fill(valid_weight <= 0.5, 0.0).amax().detach(),
        "raw_spike_loss_before_gate": (raw_spike.sum() / valid_count.clamp_min(1.0)).detach(),
        "gated_spike_loss_after_gate": (weighted.sum() / valid_count.clamp_min(1.0)).detach(),
        "boundary_mean": ((boundary * valid_weight).sum() / valid_count.clamp_min(1.0)).detach(),
    }
    valid_gate = gate[valid_weight > 0.5]
    if valid_gate.numel() > 0:
        stats["gate_p50"] = torch.quantile(valid_gate.float(), 0.50).detach()
        stats["gate_p95"] = torch.quantile(valid_gate.float(), 0.95).detach()
        stats["gate_p99"] = torch.quantile(valid_gate.float(), 0.99).detach()
    return loss, stats
