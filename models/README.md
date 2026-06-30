# 预训练模型占位目录

当前 V1.0 同时支持模型直接推理和已有结果分析。DFUN 与 EviField 的
研究代码及参数由 `inference/adapters.py` 延迟加载，未安装 PyTorch 时基础
结果分析功能仍可使用。

实际接入目录：

```text
dfun_model_package/
├── checkpoints/dfun_gate_fls_exp_test4_model.pth
├── dfun/                  # 网络、预处理和推理参考代码
└── mappings/              # 230类空间群映射

system_handoff_evifield_era/
├── checkpoints/evifield_era_optical_flow_latest_S_Mix.pt
├── code/                  # 网络、NIW变换和预处理
└── test_sample/           # 双帧测试图像
```

安装模型推理依赖：

```bash
pip install -e '.[models]'
```

不要把参数文件提交到公开代码仓库。
