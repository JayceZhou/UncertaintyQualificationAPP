"""Build copyright application and product manual from the supplied templates."""

from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_ALIGN_VERTICAL, WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK, WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Inches, Pt, RGBColor


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output" / "docs"
ASSETS = ROOT / "docs" / "assets"
APPLICATION_TEMPLATE = Path(
    "/Users/jayce/Documents/博士/专利/软著初稿-上海大学-钱老师-激光焊接全流程数据库管理系统/"
    "初稿-上大-钱老师-1-申请表-20260507.docx"
)
APP_NAME = "基于变分证据采样的异构材料数据不确定性分析系统"
VERSION = "V1.0"


def set_cn_font(run, name: str, size: float, bold: bool = False, color: str | None = None) -> None:
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor.from_string(color)


def set_cell_shading(cell, fill: str) -> None:
    tc_pr = cell._tc.get_or_add_tcPr()
    shading = tc_pr.find(qn("w:shd"))
    if shading is None:
        shading = OxmlElement("w:shd")
        tc_pr.append(shading)
    shading.set(qn("w:fill"), fill)


def set_cell_margins(cell, top=100, start=120, bottom=100, end=120) -> None:
    tc = cell._tc
    tc_pr = tc.get_or_add_tcPr()
    tc_mar = tc_pr.first_child_found_in("w:tcMar")
    if tc_mar is None:
        tc_mar = OxmlElement("w:tcMar")
        tc_pr.append(tc_mar)
    for margin, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        node = tc_mar.find(qn(f"w:{margin}"))
        if node is None:
            node = OxmlElement(f"w:{margin}")
            tc_mar.append(node)
        node.set(qn("w:w"), str(value))
        node.set(qn("w:type"), "dxa")


def add_page_number(paragraph) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    run._r.extend([begin, instruction, separate, text, end])
    set_cn_font(run, "Arial", 9)


def add_bottom_border(paragraph, color="808080", size="6") -> None:
    p_pr = paragraph._p.get_or_add_pPr()
    borders = p_pr.find(qn("w:pBdr"))
    if borders is None:
        borders = OxmlElement("w:pBdr")
        p_pr.append(borders)
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"), "single")
    bottom.set(qn("w:sz"), size)
    bottom.set(qn("w:space"), "4")
    bottom.set(qn("w:color"), color)
    borders.append(bottom)


def configure_document(document: Document) -> None:
    section = document.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.top_margin = Cm(2.0)
    section.bottom_margin = Cm(1.8)
    section.left_margin = Cm(2.5)
    section.right_margin = Cm(2.5)
    section.header_distance = Cm(0.8)
    section.footer_distance = Cm(0.8)

    normal = document.styles["Normal"]
    normal.font.name = "Songti SC"
    normal._element.rPr.rFonts.set(qn("w:eastAsia"), "Songti SC")
    normal.font.size = Pt(11)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(6)

    for name, size, color in (
        ("Heading 1", 18, "1F2937"),
        ("Heading 2", 15, "243B53"),
        ("Heading 3", 12, "334E68"),
    ):
        style = document.styles[name]
        style.font.name = "Heiti SC"
        style._element.rPr.rFonts.set(qn("w:eastAsia"), "Heiti SC")
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(12)
        style.paragraph_format.space_after = Pt(8)
        style.paragraph_format.keep_with_next = True

    header = section.header
    table = header.add_table(rows=1, cols=2, width=Cm(16.0))
    table.autofit = False
    table.columns[0].width = Cm(14.3)
    table.columns[1].width = Cm(1.7)
    left = table.cell(0, 0).paragraphs[0]
    left.alignment = WD_ALIGN_PARAGRAPH.LEFT
    set_cn_font(left.add_run(f"{APP_NAME} {VERSION}"), "Songti SC", 9)
    right = table.cell(0, 1).paragraphs[0]
    right.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    add_page_number(right)
    for cell in table.rows[0].cells:
        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
        set_cell_margins(cell, 0, 0, 0, 0)
    add_bottom_border(left)
    add_bottom_border(right)


def add_body(document: Document, text: str, first_indent=True) -> None:
    paragraph = document.add_paragraph(style="Normal")
    paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    if first_indent:
        paragraph.paragraph_format.first_line_indent = Pt(22)
    set_cn_font(paragraph.add_run(text), "Songti SC", 11)


def add_bullets(document: Document, items: list[str]) -> None:
    for item in items:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.left_indent = Cm(0.7)
        paragraph.paragraph_format.first_line_indent = Cm(-0.35)
        paragraph.paragraph_format.space_after = Pt(4)
        set_cn_font(paragraph.add_run(item), "Songti SC", 11)


def add_steps(document: Document, items: list[str]) -> None:
    for item in items:
        paragraph = document.add_paragraph(style="List Number")
        paragraph.paragraph_format.left_indent = Cm(0.8)
        paragraph.paragraph_format.first_line_indent = Cm(-0.4)
        paragraph.paragraph_format.space_after = Pt(5)
        set_cn_font(paragraph.add_run(item), "Songti SC", 11)


def add_table(document: Document, headers: list[str], rows: list[list[str]], widths: list[float]) -> None:
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = False
    table.style = "Table Grid"
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.width = Cm(widths[index])
        set_cell_shading(cell, "DCE6F1")
        cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
        paragraph = cell.paragraphs[0]
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        set_cn_font(paragraph.add_run(header), "Heiti SC", 10, bold=True)
        set_cell_margins(cell)
    for row_values in rows:
        cells = table.add_row().cells
        for index, value in enumerate(row_values):
            cells[index].width = Cm(widths[index])
            cells[index].vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            paragraph = cells[index].paragraphs[0]
            paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if index == len(headers) - 1 else WD_ALIGN_PARAGRAPH.CENTER
            set_cn_font(paragraph.add_run(value), "Songti SC", 9.5)
            set_cell_margins(cells[index])
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def add_screenshot(document: Document, filename: str, caption: str) -> None:
    path = ASSETS / filename
    if not path.is_file():
        return
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.paragraph_format.keep_with_next = True
    paragraph.add_run().add_picture(str(path), width=Cm(15.5))
    caption_paragraph = document.add_paragraph()
    caption_paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    caption_paragraph.paragraph_format.space_after = Pt(8)
    set_cn_font(caption_paragraph.add_run(caption), "Songti SC", 9.5)


def page_break(document: Document) -> None:
    # The source template uses natural page flow. Explicit break paragraphs can
    # be pushed to a new page by a preceding table and create blank pages.
    return None


def build_manual() -> Path:
    document = Document()
    configure_document(document)

    document.add_heading("一、软件概述", level=1)
    document.add_heading("1.1 软件简介", level=2)
    add_body(
        document,
        "本系统面向粉末衍射分类结果和二维材料矢量场预测结果，提供统一的不确定性量化、风险分级、"
        "高风险对象筛选与可视化分析能力。系统将 DFUN 工作中的 Monte Carlo Dropout 多次随机前向"
        "统计方法，与 EviField 工作中的二维 Normal-Inverse-Wishart（NIW）证据协方差方法集成到同一"
        "轻量化网页界面中，用于辅助研究人员判断模型预测是否需要人工复核。",
    )
    add_body(
        document,
        "V1.0 优先实现结果分析模式：用户上传模型已有推理结果，系统完成数值校验、指标计算、风险排序、"
        "图形展示和 CSV 导出，并提供用户注册登录、项目空间、用户状态管理和操作审计。软件不承担神经"
        "网络在线训练或实验设备控制。DFUN 和"
        "EviField 的预训练参数加载目录与推理适配器已经预留，待参数文件、网络结构和预处理配置齐备后"
        "即可接入模型推理模式。",
    )
    document.add_page_break()
    document.add_heading("1.2 系统架构", level=2)
    add_table(
        document,
        ["层次", "组成", "职责"],
        [
            ["交互层", "Streamlit", "注册登录、项目管理、文件上传、结果展示、风险筛选与下载"],
            ["分析层", "分类 UQ / 2D NIW / 风险评分", "计算预测熵、互信息、协方差分解、几何诊断和风险分数"],
            ["模型层", "DFUN / EviField 适配器", "当前保留参数加载与规范化输出接口，待模型文件上传后启用"],
            ["数据层", "SQLite / CSV / 内存数据表", "保存账户、项目和审计日志；接收推理结果并生成分析表"],
        ],
        [2.2, 4.2, 9.0],
    )
    add_body(document, "该结构将模型推理与不确定性分析解耦。即使模型参数尚未接入，用户仍可对已有推理结果执行全部核心分析。")

    document.add_page_break()
    document.add_heading("1.3 软件运行环境", level=2)
    add_table(
        document,
        ["组件", "最低要求", "推荐配置"],
        [
            ["操作系统", "Windows 10 / macOS 12 / Ubuntu 20.04", "64 位桌面或服务器系统"],
            ["Python", "3.10", "3.11 或 3.12"],
            ["内存", "8 GB", "16 GB 以上"],
            ["磁盘", "2 GB 可用空间", "10 GB 以上（不含模型参数）"],
            ["浏览器", "Chrome / Edge / Firefox 最新版", "分辨率 1280×720 以上"],
            ["GPU", "结果分析模式不要求", "模型推理模式按参数文件要求配置"],
        ],
        [3.0, 5.6, 6.8],
    )
    document.add_heading("1.4 软件功能概述", level=2)
    add_bullets(
        document,
        [
            "用户与权限管理：支持注册、登录、退出和密码修改；首个账户为管理员，后续账户为分析用户。",
            "项目与审计管理：按任务建立分析项目，管理员可管理用户状态并查看操作日志。",
            "分类不确定性分析：汇总同一样本的多次类别概率，计算预测类别、Top-k、预测熵、互信息、概率方差和采样分歧。",
            "证据协方差分析：从二维 NIW 参数计算偶然、认知及总预测协方差，并生成迹、特征值、各向异性、相关性与椭圆参数。",
            "统一风险分级：将不同任务的不确定性指标组合为 0 至 100 的排序分数，并划分低、中、高和严重风险。",
            "风险覆盖分析：在提供真实标签或真实矢量时，计算保留不同覆盖率时的错误风险。",
            "结果筛选与导出：按风险分数筛选需复核对象，并将完整结果或高风险清单导出为 CSV。",
            "预训练模型管理：展示 DFUN 与 EviField 参数文件状态，提供稳定的后续接入路径。",
        ],
    )

    page_break(document)
    document.add_heading("二、软件使用说明", level=1)
    document.add_heading("2.1 启动软件", level=2)
    add_steps(
        document,
        [
            "进入软件根目录并创建 Python 虚拟环境。",
            "执行 pip install -e . 安装 Streamlit、NumPy、pandas 与 Plotly。",
            "执行 streamlit run app.py 启动本地服务。",
            "在浏览器打开命令行显示的本地地址，进入账户登录页面。",
        ],
    )
    document.add_heading("2.1.1 注册与登录", level=3)
    add_body(
        document,
        "首次运行时切换到“创建账户”，填写显示名称、用户名和密码。首个注册账户自动成为系统管理员，"
        "后续注册账户默认为分析用户。密码至少八位并同时包含字母和数字，数据库只保存 scrypt 加盐哈希。",
    )
    add_screenshot(document, "login-v2.png", "图2.1 用户登录页面")
    document.add_heading("2.1.2 系统总览", level=3)
    add_screenshot(document, "dashboard-v2.png", "图2.2 系统总览页面")
    add_body(document, "系统总览展示项目数、有效用户、个人操作记录、模型状态和核心分析能力。管理员侧栏额外显示用户管理与操作日志。")

    document.add_heading("2.2 项目与用户基础管理", level=2)
    add_body(
        document,
        "分析用户可在项目空间新建项目，记录项目名称、编号、任务类型和说明，并仅查看自己的项目。系统"
        "管理员可查看全部项目，在用户管理页面启用或停用普通账户，并通过操作日志审计注册、登录、项目"
        "变更、分析执行和密码修改等操作。管理员不能停用当前登录账户。",
    )

    page_break(document)
    document.add_heading("2.3 分类不确定性分析", level=2)
    document.add_heading("2.3.1 输入文件", level=3)
    add_body(
        document,
        "分类模块读取长表 CSV。必要字段为 sample_id、pass_id、class_label 和 probability，可选字段 true_label。"
        "同一样本必须具有相同采样次数，每次采样必须包含相同类别集合。系统允许概率行和不等于 1，并在验证通过后自动归一化。",
    )
    add_table(
        document,
        ["字段", "含义", "约束"],
        [
            ["sample_id", "样本唯一编号", "同一样本可出现多行"],
            ["pass_id", "随机前向传播轮次", "各样本轮次数一致"],
            ["class_label", "类别名称或编号", "每轮类别集合一致"],
            ["probability", "当前类别概率", "有限且非负"],
            ["true_label", "真实类别", "可选，用于准确率、ECE 和风险曲线"],
        ],
        [3.0, 6.0, 6.4],
    )
    document.add_heading("2.3.2 执行分析", level=3)
    add_steps(
        document,
        [
            "点击“分类不确定性”，上传符合格式的 CSV；也可点击“载入演示数据”。",
            "展开输入预览检查字段和值域。",
            "点击“执行分类分析”，等待页面显示统计卡片和图表。",
            "按 risk_score 降序检查高风险样本，必要时下载完整结果。",
        ],
    )

    page_break(document)
    document.add_heading("2.3.3 分类结果说明", level=3)
    add_screenshot(document, "classification-analysis.png", "图2.3 分类不确定性分析页面")
    add_table(
        document,
        ["指标", "解释"],
        [
            ["confidence", "平均概率中最大值，越大表示最终类别更集中"],
            ["predictive_entropy", "平均预测分布的熵，越大表示类别整体越模糊"],
            ["mutual_information", "预测熵减去采样期望熵，用于描述多次随机推理之间的分歧"],
            ["variation_ratio", "多次推理的获胜类别不一致比例"],
            ["risk_score", "0 至 100 的综合排序分数，不解释为真实错误概率"],
        ],
        [5.0, 10.4],
    )

    page_break(document)
    document.add_heading("2.4 二维矢量场证据协方差分析", level=2)
    document.add_heading("2.4.1 输入文件", level=3)
    add_body(
        document,
        "二维矢量场模块读取像素级 NIW 参数 CSV。必要字段为 x、y、mean_1、mean_2、kappa、nu、l11、l21、l22。"
        "其中 l11、l21、l22 构成尺度矩阵的下三角 Cholesky 因子；kappa 必须大于 0，nu 必须大于 3，"
        "l11 与 l22 必须大于 0。可选 target_1 和 target_2 用于端点误差与风险覆盖分析。",
    )
    add_steps(
        document,
        [
            "点击“矢量场不确定性”，上传像素级 CSV；也可使用演示矢量场。",
            "点击“执行矢量场分析”计算协方差分解和几何诊断。",
            "在热力图下拉框中切换统一风险、协方差迹、各向异性、相关性和证据风险。",
            "检查像素级结果并导出 CSV。",
        ],
    )
    add_body(document, "软件以输入坐标 x、y 重构二维热力图。若坐标不构成规则网格，表格结果仍可正常计算，但热力图可能包含空白位置。")

    page_break(document)
    document.add_heading("2.4.2 矢量场结果说明", level=3)
    add_screenshot(document, "vector-analysis.png", "图2.4 二维矢量场证据协方差分析页面")
    add_table(
        document,
        ["指标", "解释"],
        [
            ["trace_uncertainty", "总预测协方差迹的一半，描述局部总体不确定性尺度"],
            ["anisotropy_ratio", "最大与最小特征值之比，描述方向不均衡程度"],
            ["correlation", "两个矢量分量预测误差的归一化相关性"],
            ["principal_angle_deg", "最大特征值对应的主不确定方向"],
            ["ellipse_major/minor_95", "95% 椭圆的长轴和短轴直径"],
            ["evidence_risk", "由 kappa 与 nu 得到的证据不足分数，值越大风险越高"],
        ],
        [5.0, 10.4],
    )

    page_break(document)
    document.add_heading("2.5 高风险筛选与导出", level=2)
    add_body(
        document,
        "完成任一分析后进入“高风险筛选”页面。风险阈值滑块默认设置为 60，页面仅显示不低于该阈值的"
        "样本或像素。列表包含任务类型、对象编号、风险分数、风险等级、主要风险原因和复核建议。",
    )
    add_bullets(
        document,
        [
            "低风险：0 ≤ 分数 < 30；",
            "中风险：30 ≤ 分数 < 60；",
            "高风险：60 ≤ 分数 < 80；",
            "严重风险：80 ≤ 分数 ≤ 100。",
        ],
    )
    add_body(
        document,
        "风险分数用于排序、筛选和分配人工复核资源，不是模型预测错误概率。没有真实标签时，软件只输出"
        "相对风险；存在真实标签时，Risk-Coverage 曲线用于检验风险排序是否能优先剔除错误对象。",
    )
    document.add_heading("2.6 模型管理", level=2)
    add_body(
        document,
        "模型管理页面检查 models/dfun/checkpoint.pt 与 models/evifield/checkpoint.pt 是否存在。当前参数文件"
        "尚未上传，页面显示“等待上传”。参数接入后，inference/adapters.py 将负责将模型输出转换为两类"
        "分析内核所需的标准字段。",
    )

    page_break(document)
    document.add_heading("三、核心算法说明", level=1)
    document.add_heading("3.1 Monte Carlo Dropout 分类量化", level=2)
    add_body(
        document,
        "对每个样本执行 T 次启用 Dropout 的随机前向传播，得到类别概率 p_t。软件先计算平均概率 p̄，"
        "再取最大概率对应类别为最终预测。预测熵为 H(p̄) = -Σ p̄_k ln(p̄_k)。采样期望熵为各次预测熵"
        "的平均值，互信息为 MI = H(p̄) - E[H(p_t)]。该过程对应 DFUN 论文采用 T=50 次前向传播评估"
        "预测均值、方差与熵的思路。",
    )
    add_body(
        document,
        "分类风险分数由归一化预测熵、1-confidence、平均概率方差和 variation ratio 按固定权重组合："
        "Risk_cls = 100 × (0.40H_norm + 0.30(1-c) + 0.15V_norm + 0.15R_var)。所有分量限制在 0 至 1。",
    )
    document.add_heading("3.2 风险覆盖", level=2)
    add_body(
        document,
        "软件按不确定性从低到高排序对象，依次增加保留数量。Coverage 为已保留对象占全部对象的比例；"
        "Risk 为保留对象的累计平均错误。理想风险排序应使低覆盖率区域保持较低风险，并随覆盖率上升而"
        "逐步接近全体数据风险。",
    )

    page_break(document)
    document.add_heading("3.3 二维 NIW 协方差分解", level=2)
    add_body(
        document,
        "EviField 在每个像素输出均值向量 m、证据参数 κ、自由度 ν 以及正定尺度矩阵 Ψ。软件通过"
        "L=[[l11,0],[l21,l22]] 与 Ψ=LLᵀ+εI 保证尺度矩阵正定。二维条件下 ν>3。",
    )
    add_bullets(
        document,
        [
            "偶然协方差：C_ale = Ψ / (ν - 3)；",
            "认知协方差：C_epi = Ψ / [κ(ν - 3)]；",
            "总预测协方差：C_tot = (κ + 1)Ψ / [κ(ν - 3)]；",
            "证据风险：s_E = -ln(1 + κ) - ln(1 + ν - 3 + ε)。",
        ],
    )
    add_body(
        document,
        "对 C_tot 进行对称特征分解得到最大、最小特征值与主特征向量。软件进一步计算迹、特征值间隙、"
        "各向异性比、通道相关系数、条件数、主方向角和 95% 椭圆轴长。NIW 所给出的偶然/认知分解应"
        "理解为模型内部诊断量，不等同于严格可证明的因果不确定性分解。",
    )

    page_break(document)
    document.add_heading("四、数据校验与异常处理", level=1)
    document.add_heading("4.1 分类数据校验", level=2)
    add_bullets(
        document,
        [
            "缺少必要字段时终止分析并列出缺失字段；",
            "概率为负数、空值或无穷值时拒绝输入；",
            "同一样本、轮次和类别出现重复记录时拒绝输入；",
            "各轮类别集合不一致或样本采样次数不一致时拒绝输入；",
            "每轮概率总和大于 0 时自动归一化。",
        ],
    )
    document.add_heading("4.2 NIW 数据校验", level=2)
    add_bullets(
        document,
        [
            "全部参数必须为有限数值；",
            "κ≤0、ν≤3 或 Cholesky 对角元素非正时拒绝输入；",
            "可选真实矢量必须同时包含 target_1 与 target_2；",
            "特征分解使用对称矩阵算法，并添加 εI 提高数值稳定性。",
        ],
    )
    document.add_heading("4.3 测试结果", level=2)
    add_body(
        document,
        "当前版本包含分类确定性样本、字段缺失、NIW 协方差公式和自由度约束等单元测试。测试同时验证"
        "分类互信息在完全一致的重复预测下为零，以及二维 NIW 偶然、认知和总协方差满足公式关系。",
    )

    page_break(document)
    document.add_heading("五、预训练模型接入说明", level=1)
    document.add_heading("5.1 DFUN 参数占位", level=2)
    add_body(
        document,
        "DFUN 参数文件约定放置于 models/dfun/checkpoint.pt。接入时还需提供网络结构代码、类别编号映射、"
        "衍射曲线长度、归一化参数、峰值特征生成方式和 Dropout 层配置。适配器的标准输出为长表格式的"
        "sample_id、pass_id、class_label 与 probability。",
    )
    document.add_heading("5.2 EviField 参数占位", level=2)
    add_body(
        document,
        "EviField 参数文件约定放置于 models/evifield/checkpoint.pt。接入时还需提供网络结构代码、输入图像"
        "通道定义、尺寸与归一化方式，以及输出张量中 mean_1、mean_2、kappa、nu、l11、l21、l22 的"
        "通道顺序。适配器输出像素级标准表后直接调用现有 NIW 分析内核。",
    )
    document.add_heading("5.3 版本边界", level=2)
    add_body(
        document,
        "在预训练参数尚未安装时，模型管理页面仅显示占位状态，不允许伪造推理结果。结果分析模式不依赖"
        "PyTorch，可在 CPU 环境运行；模型推理模式的 PyTorch、CUDA 和显存要求将在收到参数文件后按"
        "实际模型补充。",
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "产品文档-异构材料数据不确定性分析系统-V1.0.docx"
    document.save(path)
    return path


def replace_cell_text(cell, text: str) -> None:
    first_paragraph = cell.paragraphs[0]
    if first_paragraph.runs:
        first_paragraph.runs[0].text = text
        for run in first_paragraph.runs[1:]:
            run.text = ""
    else:
        first_paragraph.add_run(text)
    for paragraph in cell.paragraphs[1:]:
        for run in paragraph.runs:
            run.text = ""


def build_application() -> Path:
    document = Document(APPLICATION_TEMPLATE)
    table = document.tables[0]
    replacements = {
        (1, 4): "SciUQ Studio",
        (14, 8): "PC机，4核CPU，16G内存，20G可用硬盘",
        (15, 8): "PC或服务器，4核CPU，8G内存，10G硬盘，GPU可选",
        (16, 11): "macOS 12及以上或Ubuntu 20.04",
        (17, 11): "Python3.10+，Streamlit，SQLite，NumPy，pandas，Plotly",
        (18, 11): "Windows 10、macOS 12或Ubuntu 20.04及以上",
        (19, 11): "Python3.10+，Streamlit，SQLite3，现代浏览器；PyTorch可选",
        (20, 4): "Python3.10及以上",
        (20, 12): "1526行",
        (23, 9): (
            "支持用户注册登录、项目管理和日志审计；导入多次分类概率或二维NIW参数，计算预测熵、"
            "互信息、协方差分解及证据风险，实现0-100风险分级、高风险筛选、可视化和CSV导出。"
        ),
        (25, 6): (
            "采用Streamlit和SQLite轻量架构，scrypt加盐保护密码，NumPy向量化计算、Plotly展示风险图表；"
            "模型推理与分析内核解耦，预留DFUN和EviField参数接口。"
        ),
    }
    for (row_index, cell_index), value in replacements.items():
        replace_cell_text(table.rows[row_index].cells[cell_index], value)
    # This template cell contains a highlighted hyperlink-like fragment that is
    # not exposed as a normal python-docx run. Rebuild the cell to remove it.
    table.rows[25].cells[6].text = replacements[(25, 6)]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT / "软著申请表-异构材料数据不确定性分析系统-V1.0-草稿.docx"
    document.save(path)
    return path


if __name__ == "__main__":
    print(build_manual())
    print(build_application())
