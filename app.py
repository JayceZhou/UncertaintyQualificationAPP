"""SciUQ Studio V1.0 - authenticated Streamlit application."""

from dataclasses import asdict
import html
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from appcore import AppStore, User
from ui import inject_theme, page_header, section_label
from uqcore import analyze_mc_probabilities, analyze_niw_field


APP_NAME = "基于变分证据采样的异构材料数据不确定性分析系统"
APP_SHORT_NAME = "SciUQ Studio"
ROOT = Path(__file__).resolve().parent
RISK_COLORS = {
    "低风险": "#10b981",
    "中风险": "#f59e0b",
    "高风险": "#f97316",
    "严重风险": "#ef4444",
}


@st.cache_resource
def get_store(path: str) -> AppStore:
    return AppStore(path)


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


def style_figure(figure, height: int = 390):
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=25, r=20, t=55, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#ffffff",
        font=dict(family="Inter, PingFang SC, Microsoft YaHei", color="#334155"),
        title_font=dict(size=16, color="#172033"),
        legend_title_text="",
    )
    return figure


def metric_row(items: list[tuple[str, object, str | None]]) -> None:
    columns = st.columns(len(items))
    for column, (label, value, help_text) in zip(columns, items):
        column.metric(label, value, help=help_text)


def current_user(store: AppStore) -> User | None:
    user_id = st.session_state.get("user_id")
    if user_id is None:
        return None
    user = store.get_user(int(user_id))
    if user is None or user.status != "active":
        st.session_state.pop("user_id", None)
        return None
    return user


def auth_page(store: AppStore) -> None:
    st.markdown('<div class="auth-shell"></div>', unsafe_allow_html=True)
    left, right = st.columns([1.12, 0.88], gap="large")
    with left:
        st.markdown(
            """
            <div class="auth-hero">
              <div class="auth-hero__logo">UQ</div>
              <div class="auth-hero__title">让材料智能模型的每一次预测，都有可信度依据。</div>
              <div class="auth-hero__copy">
                集成 MC Dropout 分类不确定性与二维 NIW 证据协方差分析，
                在统一工作台中完成风险量化、结果筛选与复核决策。
              </div>
              <div class="auth-list">
                <div><span>01</span>多次随机预测的熵、互信息与稳定性分析</div>
                <div><span>02</span>二维矢量场全协方差与方向耦合诊断</div>
                <div><span>03</span>0-100 风险分级与高风险对象优先复核</div>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        st.markdown('<div class="auth-panel-title">欢迎使用 SciUQ</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-panel-copy">登录账户或创建新的分析账户</div>', unsafe_allow_html=True)
        login_tab, register_tab = st.tabs(["账户登录", "创建账户"])
        with login_tab:
            with st.form("login_form"):
                username = st.text_input("用户名", placeholder="输入用户名", key="login_username")
                password = st.text_input("密码", type="password", placeholder="输入密码", key="login_password")
                submitted = st.form_submit_button("登录系统", type="primary", width="stretch")
            if submitted:
                user = store.authenticate(username, password)
                if user is None:
                    store.log(None, "login_failed", "user", details={"username": username.strip()})
                    st.error("用户名、密码错误或账户已停用")
                else:
                    store.log(user.id, "login", "session", str(user.id))
                    st.session_state.user_id = user.id
                    st.rerun()
        with register_tab:
            first_account = len(store.list_users()) == 0
            if first_account:
                st.info("系统尚无账户，首个注册账户将自动成为管理员。")
            with st.form("register_form"):
                display_name = st.text_input("显示名称", placeholder="例如：材料分析员", key="register_display")
                new_username = st.text_input("用户名", placeholder="3-32 位中文、字母、数字或下划线", key="register_username")
                new_password = st.text_input("设置密码", type="password", placeholder="至少 8 位，包含字母和数字", key="register_password")
                confirmation = st.text_input("确认密码", type="password", placeholder="再次输入密码", key="register_confirm")
                registered = st.form_submit_button("创建并登录", type="primary", width="stretch")
            if registered:
                if new_password != confirmation:
                    st.error("两次输入的密码不一致")
                else:
                    try:
                        user = store.register(new_username, display_name, new_password)
                    except ValueError as error:
                        st.error(str(error))
                    else:
                        st.session_state.user_id = user.id
                        st.rerun()
        st.caption("密码采用 scrypt 加盐哈希保存；系统不会保存明文密码。")


def sidebar(user: User, store: AppStore) -> str:
    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
          <span class="sidebar-brand__mark">UQ</span><span class="sidebar-brand__name">SciUQ Studio</span>
          <div class="sidebar-brand__sub">MATERIAL INTELLIGENCE · V1.0</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    role_name = "系统管理员" if user.role == "admin" else "分析用户"
    st.sidebar.markdown(
        f'<div class="user-card"><div class="user-card__name">{html.escape(user.display_name)}</div>'
        f'<div class="user-card__meta"><span class="status-dot"></span>{html.escape(role_name)} · @{html.escape(user.username)}</div></div>',
        unsafe_allow_html=True,
    )
    pages = ["系统总览", "项目空间", "分类不确定性", "矢量场不确定性", "高风险筛选", "模型管理", "账户设置"]
    if user.role == "admin":
        pages.extend(["用户管理", "操作日志"])
    selected = st.sidebar.radio("工作区导航", pages, label_visibility="collapsed")
    st.sidebar.divider()
    if st.sidebar.button("退出登录", width="stretch"):
        store.log(user.id, "logout", "session", str(user.id))
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    st.sidebar.caption("© 2026 SciUQ · 上海大学")
    return selected


def home_page(user: User, store: AppStore) -> None:
    stats = store.dashboard_stats(user)
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero__eyebrow">UNCERTAINTY INTELLIGENCE WORKSPACE</div>
          <div class="hero__title">你好，{html.escape(user.display_name)}。从预测结果走向可复核的科学判断。</div>
          <div class="hero__copy">统一分析分类概率与二维证据协方差，识别模型不确定区域，优先安排高风险样本复核。</div>
          <div class="hero__pill"><span class="status-dot"></span>分析服务运行正常</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    metric_row(
        [
            ("分析项目", stats["projects"], "管理员可查看系统全部项目"),
            ("有效用户", stats["active_users"], None),
            ("我的操作记录", stats["operations"], None),
            ("已接入模型", 0, "模型参数上传后自动更新"),
        ]
    )
    section_label("核心分析能力")
    cards = st.columns(3, gap="medium")
    content = [
        ("◫", "分类不确定性", "汇总 MC Dropout 多次预测，计算熵、互信息、Top-k 和风险覆盖。", "适用于 DFUN / 空间群分类"),
        ("⌁", "证据协方差", "解析二维 NIW 参数，诊断偶然、认知、方向耦合与不确定性椭圆。", "适用于 EviField / 矢量场"),
        ("◇", "风险决策", "将异构指标映射为统一风险分数，筛选高风险样本和像素。", "支持人工复核工作流"),
    ]
    for column, (icon, title, copy, tag) in zip(cards, content):
        column.markdown(
            f'<div class="feature-card"><div class="feature-card__icon">{icon}</div>'
            f'<div class="feature-card__title">{title}</div><div class="feature-card__copy">{copy}</div>'
            f'<div class="feature-card__tag">{tag}</div></div>',
            unsafe_allow_html=True,
        )
    projects = store.list_projects(user)
    section_label("最近项目")
    if not projects:
        st.info("尚未创建项目。进入“项目空间”建立第一个分析项目。")
    else:
        frame = pd.DataFrame(
            [
                {
                    "项目编号": project.code,
                    "项目名称": project.name,
                    "任务类型": "分类不确定性" if project.task_type == "classification" else "二维矢量场",
                    "状态": "草稿" if project.status == "draft" else "已完成",
                    "创建时间": project.created_at,
                }
                for project in projects[:6]
            ]
        )
        st.dataframe(frame, width="stretch", hide_index=True)


def project_page(user: User, store: AppStore) -> None:
    page_header("PROJECT WORKSPACE", "项目空间", "为每次不确定性分析建立独立项目，保留任务类型、项目编号和用途说明。")
    with st.expander("新建分析项目", expanded=not bool(store.list_projects(user))):
        with st.form("create_project_form"):
            left, right = st.columns(2)
            name = left.text_input("项目名称", placeholder="例如：opXRD 空间群风险分析")
            code = right.text_input("项目编号", placeholder="例如：XRD-2026-001")
            task_label = st.selectbox("任务类型", ["分类不确定性分析", "二维矢量场不确定性分析"])
            description = st.text_area("项目说明", placeholder="记录数据来源、模型版本或分析目的", height=90)
            submitted = st.form_submit_button("创建项目", type="primary")
        if submitted:
            try:
                store.create_project(
                    user,
                    name,
                    code,
                    "classification" if task_label.startswith("分类") else "vector_field",
                    description,
                )
            except (ValueError, PermissionError) as error:
                st.error(str(error))
            else:
                st.success("项目已创建")
                st.rerun()

    projects = store.list_projects(user)
    section_label(f"项目列表 · {len(projects)}")
    if not projects:
        st.info("暂无项目记录。")
        return
    frame = pd.DataFrame(
        [
            {
                "ID": project.id,
                "项目编号": project.code,
                "项目名称": project.name,
                "任务类型": "分类不确定性" if project.task_type == "classification" else "二维矢量场",
                "状态": "草稿" if project.status == "draft" else "已完成",
                "说明": project.description,
                "创建时间": project.created_at,
            }
            for project in projects
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)
    with st.expander("删除项目"):
        project_map = {project.id: f"{project.code} · {project.name}" for project in projects}
        project_id = st.selectbox("选择项目", list(project_map), format_func=project_map.get)
        st.warning("删除后项目记录无法恢复，当前分析内存结果不受影响。")
        if st.button("确认删除项目"):
            try:
                store.delete_project(user, int(project_id))
            except (ValueError, PermissionError) as error:
                st.error(str(error))
            else:
                st.success("项目已删除")
                st.rerun()


def classification_page(user: User, store: AppStore) -> None:
    page_header("DFUN · MONTE CARLO DROPOUT", "分类不确定性分析", "读取多次随机前向概率，计算预测熵、互信息、概率方差、Top-k 和风险覆盖关系。")
    with st.container(border=True):
        uploaded = st.file_uploader("上传长表 CSV", type=["csv"], key="classification_file")
        left, right = st.columns([1, 3])
        use_sample = left.button("载入演示数据", width="stretch")
        right.info("必要列：sample_id、pass_id、class_label、probability；true_label 可选。")

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        st.session_state.classification_input = sample_classification()
    data = data if data is not None else st.session_state.get("classification_input")
    if data is None:
        st.info("上传推理结果或载入演示数据后开始分析。")
        return
    with st.expander("输入数据预览"):
        st.dataframe(data.head(100), width="stretch")
    if st.button("执行分类分析", type="primary", width="stretch"):
        try:
            st.session_state.classification_result = analyze_mc_probabilities(data)
        except ValueError as error:
            st.error(str(error))
        else:
            store.log(user.id, "run_analysis", "classification", details={"rows": len(data)})

    result = st.session_state.get("classification_result")
    if result is None:
        return
    summary = result.summary
    section_label("分析摘要")
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
        figure = px.histogram(
            result.samples,
            x="risk_score",
            color="risk_level",
            color_discrete_map=RISK_COLORS,
            category_orders={"risk_level": list(RISK_COLORS)},
            nbins=20,
            title="样本风险分布",
        )
        st.plotly_chart(style_figure(figure), width="stretch")
    with chart_right:
        curve = result.risk_coverage
        if "risk" in curve:
            figure = px.line(curve, x="coverage", y="risk", title="Risk-Coverage 曲线")
            figure.update_traces(line=dict(color="#2563eb", width=3))
        else:
            figure = px.line(curve, x="coverage", y="threshold", title="覆盖率-熵阈值")
            figure.update_traces(line=dict(color="#0891b2", width=3))
        st.plotly_chart(style_figure(figure), width="stretch")
    section_label("样本级结果")
    ordered = result.samples.sort_values("risk_score", ascending=False)
    st.dataframe(ordered, width="stretch", hide_index=True)
    st.download_button(
        "导出分类结果 CSV",
        result.samples.to_csv(index=False).encode("utf-8-sig"),
        "classification_uncertainty_results.csv",
        "text/csv",
    )


def field_page(user: User, store: AppStore) -> None:
    page_header("EVIFIELD · 2D NIW", "二维矢量场证据协方差分析", "从二维 NIW 参数计算偶然、认知与总协方差，并诊断方向耦合、各向异性和证据风险。")
    with st.container(border=True):
        uploaded = st.file_uploader("上传像素级 NIW 参数 CSV", type=["csv"], key="field_file")
        left, right = st.columns([1, 3])
        use_sample = left.button("载入演示矢量场", width="stretch")
        right.info("必要列：x、y、mean_1、mean_2、kappa、nu、l11、l21、l22；target_1/2 可选。")

    data = None
    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        st.session_state.field_input = sample_niw()
    data = data if data is not None else st.session_state.get("field_input")
    if data is None:
        st.info("上传 NIW 参数或载入演示矢量场后开始分析。")
        return
    with st.expander("输入参数预览"):
        st.dataframe(data.head(100), width="stretch")
    if st.button("执行矢量场分析", type="primary", width="stretch"):
        try:
            st.session_state.field_result = analyze_niw_field(data)
        except ValueError as error:
            st.error(str(error))
        else:
            store.log(user.id, "run_analysis", "vector_field", details={"pixels": len(data)})

    result = st.session_state.get("field_result")
    if result is None:
        return
    summary = result.summary
    section_label("分析摘要")
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
    heatmap = go.Figure(data=go.Heatmap(z=grid.to_numpy(), x=grid.columns, y=grid.index, colorscale="Cividis"))
    heatmap.update_layout(title=f"{view_name} 空间分布", xaxis_title="x", yaxis_title="y")
    st.plotly_chart(style_figure(heatmap, 470), width="stretch")
    if "risk" in result.risk_coverage:
        figure = px.line(result.risk_coverage, x="coverage", y="risk", title="矢量场 Risk-Coverage 曲线")
        figure.update_traces(line=dict(color="#0891b2", width=3))
        st.plotly_chart(style_figure(figure), width="stretch")
    section_label("像素级诊断结果")
    st.dataframe(pixels.sort_values("risk_score", ascending=False), width="stretch", hide_index=True)
    st.download_button(
        "导出矢量场结果 CSV",
        pixels.to_csv(index=False).encode("utf-8-sig"),
        "vector_field_uncertainty_results.csv",
        "text/csv",
    )


def risk_page() -> None:
    page_header("RISK TRIAGE", "高风险结果筛选", "统一查看分类样本与矢量场像素风险，按阈值生成优先复核清单。")
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
        st.info("请先执行分类或矢量场分析。")
        return
    combined = pd.concat(tables, ignore_index=True)
    left, right = st.columns([2, 1])
    minimum = left.slider("最低风险分数", 0, 100, 60)
    task_filter = right.selectbox("任务类型", ["全部", "分类", "矢量场"])
    filtered = combined[combined["risk_score"] >= minimum]
    if task_filter != "全部":
        filtered = filtered[filtered["task_type"] == task_filter]
    filtered = filtered.sort_values("risk_score", ascending=False)
    metric_row([("待复核对象", len(filtered), None), ("严重风险", int((filtered["risk_level"] == "严重风险").sum()), None), ("当前阈值", minimum, None)])
    st.dataframe(filtered, width="stretch", hide_index=True)
    st.download_button("导出高风险清单", filtered.to_csv(index=False).encode("utf-8-sig"), "high_risk_review_list.csv", "text/csv")


def model_page() -> None:
    page_header("MODEL REGISTRY", "预训练模型管理", "检查论文模型参数状态并查看后续推理适配所需文件。")
    rows = []
    for name, relative, task, output in [
        ("DFUN", "models/dfun/checkpoint.pt", "空间群分类 / MC Dropout", "T×K 类别概率"),
        ("EviField", "models/evifield/checkpoint.pt", "二维矢量场 / NIW", "m, κ, ν, L"),
    ]:
        path = ROOT / relative
        rows.append({"模型": name, "任务": task, "标准输出": output, "参数路径": relative, "状态": "已安装" if path.is_file() else "等待上传"})
    st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
    left, right = st.columns(2)
    left.info("DFUN 还需类别映射、输入长度、峰值特征与归一化配置。")
    right.info("EviField 还需输入通道、图像尺寸、归一化与输出通道顺序。")
    st.code("models/dfun/checkpoint.pt\nmodels/evifield/checkpoint.pt", language="text")


def account_page(user: User, store: AppStore) -> None:
    page_header("ACCOUNT SECURITY", "账户设置", "查看账户信息并更新登录密码。")
    left, right = st.columns([1, 1.4], gap="large")
    with left:
        with st.container(border=True):
            st.subheader(user.display_name)
            st.write(f"用户名：`{user.username}`")
            st.write(f"角色：{'系统管理员' if user.role == 'admin' else '分析用户'}")
            st.write(f"状态：{'正常' if user.status == 'active' else '已停用'}")
            st.caption(f"创建时间：{user.created_at}")
    with right:
        with st.form("change_password_form", clear_on_submit=True):
            st.subheader("修改密码")
            current = st.text_input("当前密码", type="password")
            new = st.text_input("新密码", type="password", help="至少 8 位，包含字母和数字")
            confirm = st.text_input("确认新密码", type="password")
            submitted = st.form_submit_button("更新密码", type="primary")
        if submitted:
            if new != confirm:
                st.error("两次输入的新密码不一致")
            else:
                try:
                    store.change_password(user, current, new)
                except ValueError as error:
                    st.error(str(error))
                else:
                    st.success("密码已更新")


def user_admin_page(user: User, store: AppStore) -> None:
    page_header("ADMINISTRATION", "用户管理", "查看注册账户并启用或停用分析用户。管理员不能停用自身账户。")
    users = store.list_users()
    frame = pd.DataFrame(
        [
            {
                "ID": item.id,
                "用户名": item.username,
                "显示名称": item.display_name,
                "角色": "管理员" if item.role == "admin" else "分析用户",
                "状态": "正常" if item.status == "active" else "已停用",
                "创建时间": item.created_at,
            }
            for item in users
        ]
    )
    metric_row([("账户总数", len(users), None), ("有效账户", sum(item.status == "active" for item in users), None), ("管理员", sum(item.role == "admin" for item in users), None)])
    st.dataframe(frame, width="stretch", hide_index=True)
    candidates = [item for item in users if item.id != user.id]
    if candidates:
        with st.form("user_status_form"):
            mapping = {item.id: f"{item.username} · {item.display_name}" for item in candidates}
            target_id = st.selectbox("选择账户", list(mapping), format_func=mapping.get)
            status_label = st.selectbox("设置状态", ["正常", "停用"])
            submitted = st.form_submit_button("保存状态")
        if submitted:
            try:
                store.set_user_status(user, int(target_id), "active" if status_label == "正常" else "disabled")
            except (ValueError, PermissionError) as error:
                st.error(str(error))
            else:
                st.success("用户状态已更新")
                st.rerun()


def log_page(store: AppStore) -> None:
    page_header("AUDIT TRAIL", "操作日志", "审计账户注册、登录、项目变更、分析执行和安全操作。")
    logs = store.list_logs(200)
    frame = pd.DataFrame(logs)
    if frame.empty:
        st.info("暂无操作日志。")
        return
    frame = frame.rename(
        columns={
            "username": "操作用户",
            "action": "操作",
            "target_type": "对象类型",
            "target_id": "对象ID",
            "details": "详情",
            "created_at": "操作时间",
        }
    )
    st.dataframe(frame[["操作时间", "操作用户", "操作", "对象类型", "对象ID", "详情"]], width="stretch", hide_index=True)


def main() -> None:
    st.set_page_config(page_title=f"{APP_SHORT_NAME} · 材料不确定性分析", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    inject_theme()
    database_path = os.getenv("SCIUQ_DB", str(ROOT / "storage" / "sciuq.db"))
    store = get_store(database_path)
    user = current_user(store)
    if user is None:
        auth_page(store)
        return
    selected = sidebar(user, store)
    pages = {
        "系统总览": lambda: home_page(user, store),
        "项目空间": lambda: project_page(user, store),
        "分类不确定性": lambda: classification_page(user, store),
        "矢量场不确定性": lambda: field_page(user, store),
        "高风险筛选": risk_page,
        "模型管理": model_page,
        "账户设置": lambda: account_page(user, store),
        "用户管理": lambda: user_admin_page(user, store),
        "操作日志": lambda: log_page(store),
    }
    pages[selected]()


if __name__ == "__main__":
    main()
