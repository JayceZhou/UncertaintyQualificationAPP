# 基于变分证据采样的异构材料数据不确定性分析系统 V1.0

本软件将 DFUN 论文中的 Monte Carlo Dropout 分类不确定性分析，与 EviField
论文中的二维 NIW 证据协方差分析集成为一个轻量级网页工具。

V1.0 当前支持“结果分析模式”：用户上传模型输出，系统计算不确定性指标、
0-100 风险分数、风险等级和风险覆盖曲线。预训练模型推理接口已在
`inference/adapters.py` 中预留，参数文件约定见 `models/README.md`。

## 核心功能

- MC Dropout 多次分类概率汇总；
- 预测熵、互信息、概率方差、Top-k、ECE 和 Risk-Coverage；
- 2D NIW 偶然/认知/总协方差分解；
- 协方差迹、特征值、各向异性、相关性、条件数和 95% 不确定性椭圆；
- 分类与矢量场统一风险评分、高风险筛选和 CSV 导出；
- DFUN 与 EviField 预训练参数接入占位。

## 启动

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
streamlit run app.py
```

浏览器访问 Streamlit 输出的本地地址。两个分析页面均提供内置演示数据。

## 数据格式

分类 CSV 使用长表格式，必要列为：

```text
sample_id,pass_id,class_label,probability
```

可选 `true_label`。同一样本的每轮推理必须包含相同类别集合。

二维 NIW CSV 的必要列为：

```text
x,y,mean_1,mean_2,kappa,nu,l11,l21,l22
```

可选 `target_1,target_2`。其中 `l11,l21,l22` 构成 NIW 尺度矩阵的 Cholesky
因子，`kappa > 0`、`nu > 3`、`l11 > 0`、`l22 > 0`。

## 测试

```bash
python -m unittest discover -s tests -v
```

