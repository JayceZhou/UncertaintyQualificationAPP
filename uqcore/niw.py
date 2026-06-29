"""2D Normal-Inverse-Wishart uncertainty decomposition and diagnostics."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk import percentile_score, risk_level


REQUIRED_COLUMNS = {"x", "y", "mean_1", "mean_2", "kappa", "nu", "l11", "l21", "l22"}


@dataclass(frozen=True)
class NIWResult:
    pixels: pd.DataFrame
    risk_coverage: pd.DataFrame
    summary: dict[str, float | int | None]


def _risk_coverage(score: np.ndarray, errors: np.ndarray | None) -> pd.DataFrame:
    order = np.argsort(score, kind="stable")
    accepted = np.arange(1, len(order) + 1)
    frame = pd.DataFrame(
        {"coverage": accepted / len(order), "threshold": score[order], "accepted": accepted}
    )
    if errors is not None:
        frame["risk"] = np.cumsum(errors[order]) / accepted
    return frame


def analyze_niw_field(data: pd.DataFrame, epsilon: float = 1e-8) -> NIWResult:
    """Compute NIW-implied covariance components and vector diagnostics."""

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"缺少必要列: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("输入数据不能为空")

    output = data.copy()
    for column in REQUIRED_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    values = output[list(REQUIRED_COLUMNS)].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("NIW 参数必须全部为有限数值")
    if (output["kappa"] <= 0).any():
        raise ValueError("kappa 必须大于 0")
    if (output["nu"] <= 3).any():
        raise ValueError("二维 NIW 的 nu 必须大于 3")
    if (output[["l11", "l22"]] <= 0).any().any():
        raise ValueError("Cholesky 对角元素 l11 和 l22 必须大于 0")

    l11 = output["l11"].to_numpy(dtype=float)
    l21 = output["l21"].to_numpy(dtype=float)
    l22 = output["l22"].to_numpy(dtype=float)
    kappa = output["kappa"].to_numpy(dtype=float)
    nu = output["nu"].to_numpy(dtype=float)

    psi11 = l11**2 + epsilon
    psi12 = l11 * l21
    psi22 = l21**2 + l22**2 + epsilon
    denominator = nu - 3.0
    ale_factor = 1.0 / denominator
    epi_factor = 1.0 / (kappa * denominator)
    total_factor = ale_factor + epi_factor

    ale11, ale12, ale22 = psi11 * ale_factor, psi12 * ale_factor, psi22 * ale_factor
    epi11, epi12, epi22 = psi11 * epi_factor, psi12 * epi_factor, psi22 * epi_factor
    total11, total12, total22 = psi11 * total_factor, psi12 * total_factor, psi22 * total_factor

    covariance = np.empty((len(output), 2, 2), dtype=float)
    covariance[:, 0, 0] = total11
    covariance[:, 0, 1] = total12
    covariance[:, 1, 0] = total12
    covariance[:, 1, 1] = total22
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    lambda_min, lambda_max = eigenvalues[:, 0], eigenvalues[:, 1]
    principal = eigenvectors[:, :, 1]

    output = output.assign(
        psi_11=psi11,
        psi_12=psi12,
        psi_22=psi22,
        aleatoric_11=ale11,
        aleatoric_12=ale12,
        aleatoric_22=ale22,
        epistemic_11=epi11,
        epistemic_12=epi12,
        epistemic_22=epi22,
        total_11=total11,
        total_12=total12,
        total_22=total22,
        trace_uncertainty=0.5 * (total11 + total22),
        lambda_min=lambda_min,
        lambda_max=lambda_max,
        anisotropy_ratio=(lambda_max + epsilon) / (lambda_min + epsilon),
        eigenvalue_gap=(lambda_max - lambda_min) / (lambda_max + lambda_min + epsilon),
        correlation=total12 / (np.sqrt(total11 * total22) + epsilon),
        condition_number=(lambda_max + epsilon) / (lambda_min + epsilon),
        principal_angle_deg=np.degrees(np.arctan2(principal[:, 1], principal[:, 0])),
        ellipse_major_95=2.0 * np.sqrt(5.991 * lambda_max),
        ellipse_minor_95=2.0 * np.sqrt(5.991 * lambda_min),
        evidence_risk=-np.log1p(kappa) - np.log1p(nu - 3.0 + epsilon),
    )

    evidence_component = percentile_score(output["evidence_risk"].to_numpy())
    trace_component = percentile_score(output["trace_uncertainty"].to_numpy())
    lambda_component = percentile_score(output["lambda_max"].to_numpy())
    anisotropy_component = np.clip(output["eigenvalue_gap"].to_numpy(), 0.0, 1.0)
    risk_score = 100.0 * (
        0.45 * evidence_component
        + 0.30 * trace_component
        + 0.15 * lambda_component
        + 0.10 * anisotropy_component
    )
    output["risk_score"] = np.clip(risk_score, 0.0, 100.0)
    output["risk_level"] = risk_level(output["risk_score"])
    output["review_recommended"] = output["risk_score"] >= 60.0
    output["risk_reason"] = np.select(
        [
            evidence_component >= 0.80,
            trace_component >= 0.80,
            anisotropy_component >= 0.70,
        ],
        ["模型证据不足", "总协方差较大", "局部各向异性显著"],
        default="综合风险较低",
    )

    errors: np.ndarray | None = None
    if {"target_1", "target_2"}.issubset(output.columns):
        target1 = pd.to_numeric(output["target_1"], errors="coerce").to_numpy(dtype=float)
        target2 = pd.to_numeric(output["target_2"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(target1).all() or not np.isfinite(target2).all():
            raise ValueError("target_1 和 target_2 必须为有限数值")
        errors = np.hypot(output["mean_1"].to_numpy() - target1, output["mean_2"].to_numpy() - target2)
        output["endpoint_error"] = errors

    risk_frame = _risk_coverage(output["evidence_risk"].to_numpy(), errors)
    summary: dict[str, float | int | None] = {
        "pixel_count": len(output),
        "mean_trace_uncertainty": float(output["trace_uncertainty"].mean()),
        "mean_epistemic_trace": float((0.5 * (epi11 + epi22)).mean()),
        "mean_aleatoric_trace": float((0.5 * (ale11 + ale22)).mean()),
        "high_risk_count": int(output["risk_level"].isin(["高风险", "严重风险"]).sum()),
        "mean_endpoint_error": float(errors.mean()) if errors is not None else None,
    }
    return NIWResult(output, risk_frame, summary)
