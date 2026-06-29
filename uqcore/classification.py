"""Uncertainty metrics for repeated probabilistic classification predictions."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .risk import risk_level


REQUIRED_COLUMNS = {"sample_id", "pass_id", "class_label", "probability"}


@dataclass(frozen=True)
class ClassificationResult:
    samples: pd.DataFrame
    risk_coverage: pd.DataFrame
    summary: dict[str, float | int | None]


def _entropy(probabilities: np.ndarray, axis: int = -1) -> np.ndarray:
    p = np.clip(probabilities, 1e-12, 1.0)
    return -np.sum(p * np.log(p), axis=axis)


def _ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for index in range(bins):
        right_closed = index == bins - 1
        mask = (confidence >= edges[index]) & (
            (confidence <= edges[index + 1]) if right_closed else (confidence < edges[index + 1])
        )
        if mask.any():
            value += mask.mean() * abs(float(correct[mask].mean()) - float(confidence[mask].mean()))
    return float(value)


def _risk_coverage(score: np.ndarray, errors: np.ndarray | None) -> pd.DataFrame:
    order = np.argsort(score, kind="stable")
    count = len(order)
    accepted = np.arange(1, count + 1)
    frame = pd.DataFrame(
        {
            "coverage": accepted / count,
            "threshold": score[order],
            "accepted": accepted,
        }
    )
    if errors is not None:
        frame["risk"] = np.cumsum(errors[order]) / accepted
    return frame


def analyze_mc_probabilities(data: pd.DataFrame) -> ClassificationResult:
    """Analyze long-form Monte Carlo probability predictions.

    Expected columns are sample_id, pass_id, class_label and probability. An
    optional true_label column enables accuracy, ECE and retained-risk metrics.
    """

    missing = REQUIRED_COLUMNS - set(data.columns)
    if missing:
        raise ValueError(f"缺少必要列: {', '.join(sorted(missing))}")
    if data.empty:
        raise ValueError("输入数据不能为空")

    clean = data.copy()
    clean["probability"] = pd.to_numeric(clean["probability"], errors="coerce")
    if clean["probability"].isna().any() or (~np.isfinite(clean["probability"])).any():
        raise ValueError("probability 必须全部为有限数值")
    if (clean["probability"] < 0).any():
        raise ValueError("probability 不能为负数")
    if clean.duplicated(["sample_id", "pass_id", "class_label"]).any():
        raise ValueError("同一样本、推理轮次和类别不能出现重复记录")

    matrix = clean.pivot(index=["sample_id", "pass_id"], columns="class_label", values="probability")
    if matrix.isna().any().any():
        raise ValueError("每个样本的每轮推理必须包含相同的类别集合")
    totals = matrix.sum(axis=1).to_numpy(dtype=float)
    if (totals <= 0).any():
        raise ValueError("每轮推理的概率和必须大于 0")
    probabilities = matrix.to_numpy(dtype=float) / totals[:, None]
    matrix.loc[:, :] = probabilities

    pass_counts = matrix.groupby(level="sample_id", sort=False).size()
    if pass_counts.nunique() != 1:
        raise ValueError("所有样本必须具有相同数量的推理轮次")

    sample_ids = matrix.index.get_level_values("sample_id").unique().to_numpy()
    classes = matrix.columns.to_numpy()
    sample_count = len(sample_ids)
    pass_count = int(pass_counts.iloc[0])
    class_count = len(classes)
    cube = matrix.to_numpy().reshape(sample_count, pass_count, class_count)

    mean_p = cube.mean(axis=1)
    variance_p = cube.var(axis=1)
    predictive_entropy = _entropy(mean_p)
    expected_entropy = _entropy(cube).mean(axis=1)
    mutual_information = np.maximum(predictive_entropy - expected_entropy, 0.0)
    normalized_entropy = predictive_entropy / np.log(class_count) if class_count > 1 else np.zeros(sample_count)
    winner_index = mean_p.argmax(axis=1)
    pass_winners = cube.argmax(axis=2)
    winner_frequency = np.array([(row == winner).mean() for row, winner in zip(pass_winners, winner_index)])

    result = pd.DataFrame(
        {
            "sample_id": sample_ids,
            "predicted_class": classes[winner_index],
            "confidence": mean_p.max(axis=1),
            "predictive_entropy": predictive_entropy,
            "normalized_entropy": normalized_entropy,
            "expected_entropy": expected_entropy,
            "mutual_information": mutual_information,
            "variation_ratio": 1.0 - winner_frequency,
            "mean_probability_variance": variance_p.mean(axis=1),
        }
    )
    variance_score = np.clip(result["mean_probability_variance"].to_numpy() / 0.25, 0.0, 1.0)
    risk_score = 100.0 * (
        0.40 * result["normalized_entropy"].to_numpy()
        + 0.30 * (1.0 - result["confidence"].to_numpy())
        + 0.15 * variance_score
        + 0.15 * result["variation_ratio"].to_numpy()
    )
    result["risk_score"] = np.clip(risk_score, 0.0, 100.0)
    result["risk_level"] = risk_level(result["risk_score"])
    result["review_recommended"] = result["risk_score"] >= 60.0
    result["risk_reason"] = np.select(
        [
            result["normalized_entropy"] >= 0.65,
            result["confidence"] < 0.50,
            result["variation_ratio"] >= 0.30,
        ],
        ["预测熵较高", "最大类别概率较低", "多次采样预测分歧较大"],
        default="综合风险较低",
    )

    for rank in range(min(3, class_count)):
        ranked = np.argsort(-mean_p, axis=1)[:, rank]
        result[f"top{rank + 1}_class"] = classes[ranked]
        result[f"top{rank + 1}_probability"] = mean_p[np.arange(sample_count), ranked]

    correct: np.ndarray | None = None
    if "true_label" in clean.columns:
        labels_per_sample = clean.groupby("sample_id", sort=False)["true_label"].agg(
            lambda values: values.dropna().iloc[0] if not values.dropna().empty else np.nan
        )
        true_labels = labels_per_sample.reindex(sample_ids).to_numpy()
        result["true_label"] = true_labels
        labeled = pd.notna(true_labels)
        result["correct"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
        if labeled.any():
            correct_all = classes[winner_index[labeled]] == true_labels[labeled]
            result.loc[labeled, "correct"] = correct_all
            correct = correct_all.astype(float)

    risk_source = None
    if correct is not None:
        labeled_result = result[result["correct"].notna()]
        risk_source = 1.0 - labeled_result["correct"].astype(float).to_numpy()
        risk_frame = _risk_coverage(labeled_result["predictive_entropy"].to_numpy(), risk_source)
    else:
        risk_frame = _risk_coverage(result["predictive_entropy"].to_numpy(), None)

    summary: dict[str, float | int | None] = {
        "sample_count": sample_count,
        "pass_count": pass_count,
        "class_count": class_count,
        "mean_entropy": float(predictive_entropy.mean()),
        "mean_normalized_entropy": float(normalized_entropy.mean()),
        "high_risk_count": int(result["risk_level"].isin(["高风险", "严重风险"]).sum()),
        "accuracy": None,
        "ece": None,
    }
    if correct is not None:
        labeled_result = result[result["correct"].notna()]
        summary["accuracy"] = float(correct.mean())
        summary["ece"] = _ece(
            labeled_result["confidence"].to_numpy(),
            correct.astype(bool),
        )
    return ClassificationResult(result, risk_frame, summary)
