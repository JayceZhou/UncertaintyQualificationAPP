"""Streamlit entry point for 科学材料不确定性分析系统 V1.0."""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from uqcore import analyze_mc_probabilities, analyze_niw_field


APP_NAME = "基于变分证据采样的异构材料数据不确定性分析系统 V1.0"
ROOT = Path(__file__).resolve().parent


def sample_classification() -> pd.DataFrame:
    rng = np.random.default_rng(20260629)
    rows: list[dict] = []
    classes = ["P1", "P2₁/c", "Pnma", "Fm-3m"]
    for sample_index in range(18):
        target = sample_index % len(classes)
        concentration = np.full(len(classes), 0.7)
        concentration[target] = 5.5 if sample_index < 12 else 1.6
        for pass_index in range(30):
            probability = rng.dirichlet(concentration)
            for class_index, label in enumerate(classes):
                rows.append(
                    {
                        "sample_id": f"XRD-{sample_index + 1:03d}",
                        "pass_id": pass_index + 1,
                        "class_label": label,
                        "probability": probability[class_index],
                        "true_label": classes[target],
                    }
                )
    return pd.DataFrame(rows)


def sample_niw() -> pd.DataFrame:
    rng = np.random.default_rng(20260629)
    rows: list[dict] = []
    width, height = 24, 18
    for y in range(height):
        for x in range(width):
            dx, dy = x - width / 2, y - height / 2
            angle = np.arctan2(dy, dx) + np.pi / 2
            edge = np.hypot(dx / width, dy / height)
            kappa = max(0.15, 8.0 - 10.0 * edge + rng.normal(0, 0.25))
            nu = max(3.15, 12.0 - 11.0 * edge + rng.normal(0, 0.3))
            rows.append(
                {
                    "x": x,
                    "y": y,
                    "mean_1": np.cos(angle),
                    "mean_2": np.sin(angle),
                    "kappa": kappa,
                    "nu": nu,
                    "l11": 0.20 + 0.55 * edge,
                    "l21": 0.12 * np.sin(angle),
                    "l22": 0.16 + 0.38 * edge,
                    "target_1": np.cos(angle) + rng.normal(0, 0.03 + 0.15 * edge),
                    "target_2": np.sin(angle) + rng.normal(0, 0.03 + 0.15 * edge),
                }
            )
    return pd.DataFrame(rows)


def metric_row(items: list[tuple[str, object, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text)


def classification_page() -> None:
    st.header("分类不确定性分析")
    st.caption("对应 DFUN：读取 Monte Carlo Dropout 多次分类概率，计算预测熵、互信息、方差和风险覆盖关系。")
    uploaded = st.file_uploader("上传长表 CSV", type=["csv"], key="classification_file")
    left, right = st.columns([1, 3])
    use_sample = left.button("载入演示数据", width="stretch")
    right.info("必要列：sample_id、pass_id、class_label、probability；true_label 可选。")

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample or "classification_input" not in st.session_state:
        if use_sample:
            st.session_state.classification_input = sample_classification()
    data = data if data is not None else st.session_state.get("classification_input")
    if data is None:
        return

    with st.expander("输入数据预览", expanded=False):
        st.dataframe(data.head(100), width="stretch")
    if st.button("执行分类分析", type="primary", width="stretch"):
        try:
            st.session_state.classification_result = analyze_mc_probabilities(data)
        except ValueError as exc:
            st.error(str(exc))

    result = st.session_state.get("classification_result")
    if result is None:
        return
    summary = result.summary
    metric_row(
        [
            ("样本数", summary["sample_count"], None),
            ("采样次数", summary["pass_count"], None),
            ("平均归一化熵", f"{summary['mean_normalized_entropy']:.3f}", None),
            ("高风险样本", summary["high_risk_count"], None),
            ("准确率", "--" if summary["accuracy"] is None else f"{summary['accuracy']:.1%}", None),
            ("ECE", "--" if summary["ece"] is None else f"{summary['ece']:.3f}", "期望校准误差，越低越好"),
        ]
    )

    chart_left, chart_right = st.columns(2)
    with chart_left:
        fig = px.histogram(result.samples, x="risk_score", color="risk_level", nbins=20, title="样本风险分布")
        st.plotly_chart(fig, width="stretch")
    with chart_right:
        curve = result.risk_coverage
        if "risk" in curve:
            fig = px.line(curve, x="coverage", y="risk", title="Risk-Coverage 曲线")
        else:
            fig = px.line(curve, x="coverage", y="threshold", title="覆盖率-熵阈值")
        st.plotly_chart(fig, width="stretch")

    st.subheader("分析结果")
    st.dataframe(
        result.samples.sort_values("risk_score", ascending=False),
        width="stretch",
        hide_index=True,
    )
    st.download_button(
        "导出分类结果 CSV",
        result.samples.to_csv(index=False).encode("utf-8-sig"),
        "classification_uncertainty_results.csv",
        "text/csv",
    )


def field_page() -> None:
    st.header("二维矢量场证据协方差分析")
    st.caption("对应 EviField：从 2D NIW 参数计算偶然/认知/总协方差、证据风险和不确定性椭圆。")
    uploaded = st.file_uploader("上传像素级 NIW 参数 CSV", type=["csv"], key="field_file")
    left, right = st.columns([1, 3])
    use_sample = left.button("载入演示矢量场", width="stretch")
    right.info("必要列：x、y、mean_1、mean_2、kappa、nu、l11、l21、l22；真实矢量 target_1/2 可选。")

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample or "field_input" not in st.session_state:
        if use_sample:
            st.session_state.field_input = sample_niw()
    data = data if data is not None else st.session_state.get("field_input")
    if data is None:
        return
    with st.expander("输入参数预览", expanded=False):
        st.dataframe(data.head(100), width="stretch")
    if st.button("执行矢量场分析", type="primary", width="stretch"):
        try:
            st.session_state.field_result = analyze_niw_field(data)
        except ValueError as exc:
            st.error(str(exc))

    result = st.session_state.get("field_result")
    if result is None:
        return
    summary = result.summary
    metric_row(
        [
            ("有效像素", summary["pixel_count"], None),
            ("平均总不确定性", f"{summary['mean_trace_uncertainty']:.4f}", None),
            ("偶然不确定性", f"{summary['mean_aleatoric_trace']:.4f}", None),
            ("认知不确定性", f"{summary['mean_epistemic_trace']:.4f}", None),
            ("高风险像素", summary["high_risk_count"], None),
            ("平均 EPE", "--" if summary["mean_endpoint_error"] is None else f"{summary['mean_endpoint_error']:.4f}", None),
        ]
    )

    pixels = result.pixels
    view_name = st.selectbox(
        "热力图指标",
        ["risk_score", "trace_uncertainty", "anisotropy_ratio", "correlation", "evidence_risk"],
        format_func={
            "risk_score": "统一风险分数",
            "trace_uncertainty": "总协方差迹",
            "anisotropy_ratio": "各向异性",
            "correlation": "通道相关性",
            "evidence_risk": "证据风险",
        }.get,
    )
    grid = pixels.pivot(index="y", columns="x", values=view_name).sort_index(ascending=False)
    heatmap = go.Figure(data=go.Heatmap(z=grid.to_numpy(), x=grid.columns, y=grid.index, colorscale="Turbo"))
    heatmap.update_layout(title=f"{view_name} 空间分布", xaxis_title="x", yaxis_title="y")
    st.plotly_chart(heatmap, width="stretch")

    if "risk" in result.risk_coverage:
        fig = px.line(result.risk_coverage, x="coverage", y="risk", title="矢量场 Risk-Coverage 曲线")
        st.plotly_chart(fig, width="stretch")

    st.subheader("像素级诊断结果")
    st.dataframe(pixels.sort_values("risk_score", ascending=False), width="stretch", hide_index=True)
    st.download_button(
        "导出矢量场结果 CSV",
        pixels.to_csv(index=False).encode("utf-8-sig"),
        "vector_field_uncertainty_results.csv",
        "text/csv",
    )


def home_page() -> None:
    st.title(APP_NAME)
    st.write("面向粉末衍射分类和二维材料矢量场的轻量级不确定性分析工具。")
    st.info("V1.0 优先实现结果分析模式。预训练参数上传后，可通过已预留的模型适配器切换到直接推理模式。")
    cards = st.columns(3)
    cards[0].subheader("变分随机采样")
    cards[0].write("汇总 MC Dropout 多次前向结果，输出预测熵、互信息、Top-k 与风险排序。")
    cards[1].subheader("证据协方差")
    cards[1].write("解析 2D NIW 参数，输出偶然/认知协方差、方向耦合和证据风险。")
    cards[2].subheader("风险筛选")
    cards[2].write("将异构指标映射到 0-100 风险分数，标记需人工复核的样本或像素。")
    st.markdown("#### 使用流程")
    st.write("选择左侧分析模块 → 上传 CSV 或使用演示数据 → 执行分析 → 查看图表并导出结果。")


def risk_page() -> None:
    st.header("高风险结果筛选")
    tables = []
    classification = st.session_state.get("classification_result")
    if classification is not None:
        frame = classification.samples.copy()
        frame.insert(0, "task_type", "分类")
        tables.append(frame[["task_type", "sample_id", "predicted_class", "risk_score", "risk_level", "risk_reason", "review_recommended"]])
    field = st.session_state.get("field_result")
    if field is not None:
        frame = field.pixels.copy()
        frame["sample_id"] = frame.apply(lambda row: f"pixel({int(row.x)},{int(row.y)})", axis=1)
        frame.insert(0, "task_type", "矢量场")
        frame["predicted_class"] = "--"
        tables.append(frame[["task_type", "sample_id", "predicted_class", "risk_score", "risk_level", "risk_reason", "review_recommended"]])
    if not tables:
        st.warning("请先执行分类或矢量场分析。")
        return
    combined = pd.concat(tables, ignore_index=True)
    minimum = st.slider("最低风险分数", 0, 100, 60)
    filtered = combined[combined["risk_score"] >= minimum].sort_values("risk_score", ascending=False)
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button(
        "导出高风险清单",
        filtered.to_csv(index=False).encode("utf-8-sig"),
        "high_risk_review_list.csv",
        "text/csv",
    )


def model_page() -> None:
    st.header("预训练模型管理")
    st.caption("当前为参数文件占位状态；上传模型后补充 PyTorch 加载和预处理映射。")
    rows = []
    for name, relative, task in [
        ("DFUN", "models/dfun/checkpoint.pt", "空间群分类 / MC Dropout"),
        ("EviField", "models/evifield/checkpoint.pt", "二维矢量场 / NIW"),
    ]:
        path = ROOT / relative
        rows.append({"模型": name, "任务": task, "参数路径": relative, "状态": "已安装" if path.is_file() else "等待上传"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    st.code(
        "models/dfun/checkpoint.pt\nmodels/evifield/checkpoint.pt",
        language="text",
    )
    st.warning("仅有参数文件通常不足以完成接入，还需对应网络结构代码、预处理参数和输出通道定义。")


def main() -> None:
    st.set_page_config(page_title=APP_NAME, page_icon="🔬", layout="wide")
    st.sidebar.title("SciUQ Studio")
    page = st.sidebar.radio(
        "功能导航",
        ["系统首页", "分类不确定性", "矢量场不确定性", "高风险筛选", "模型管理"],
    )
    st.sidebar.caption("软件版本 V1.0")
    pages = {
        "系统首页": home_page,
        "分类不确定性": classification_page,
        "矢量场不确定性": field_page,
        "高风险筛选": risk_page,
        "模型管理": model_page,
    }
    pages[page]()


if __name__ == "__main__":
    main()
