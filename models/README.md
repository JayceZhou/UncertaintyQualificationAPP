# 预训练模型占位目录

当前 V1.0 可直接分析已有推理结果。模型推理模式将在参数文件上传后通过
`inference/adapters.py` 接入，分析内核和界面无需改动。

约定目录：

```text
models/
├── dfun/
│   ├── checkpoint.pt      # 待上传
│   └── model_config.json  # 类别表、输入长度、归一化参数等
└── evifield/
    ├── checkpoint.pt      # 待上传
    └── model_config.json  # 输入通道、输出参数映射、图像尺寸等
```

接入时还需要以下信息：

- 完整网络结构代码或可导入模块；
- 训练时使用的 PyTorch 版本；
- 输入预处理和归一化参数；
- DFUN 的类别编号映射与 Dropout 层配置；
- EviField 输出张量中 `mean_1, mean_2, kappa, nu, l11, l21, l22` 的通道顺序。

不要把参数文件提交到公开代码仓库。

