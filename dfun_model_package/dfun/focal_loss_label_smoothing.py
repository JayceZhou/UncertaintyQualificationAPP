import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLossWithLabelSmoothing(nn.Module):
    def __init__(self, gamma=1.5, label_smoothing=0.1, reduction='mean'):
        """
        一个结合了 Focal Loss 和标签平滑的损失函数。

        Args:
            gamma (float, optional): 聚焦参数。默认为 2.0。
            label_smoothing (float, optional): 标签平滑因子 (epsilon)。默认为 0.1。
            reduction (str, optional): 规约方式: 'mean' | 'sum'。默认为 'mean'。
        """
        super(FocalLossWithLabelSmoothing, self).__init__()
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs (torch.Tensor): 模型的原始输出 (logits)，形状为 (N, C)。
            targets (torch.Tensor): 真实的整数标签，形状为 (N)。
        """
        # --- 1. 计算带有标签平滑的交叉熵损失 ---
        # 我们让 PyTorch 内置的函数为我们处理标签平滑的复杂计算
        # 注意：这里的 reduction='none' 是为了得到每个样本的单独损失值，以便后续加权
        ce_loss = F.cross_entropy(
            inputs, 
            targets, 
            label_smoothing=self.label_smoothing, 
            reduction='none'
        )
        
        # --- 2. 计算 Focal Loss 的调制因子 ---
        # 首先，获取模型对每个样本的预测概率分布
        probs = F.softmax(inputs, dim=1)
        
        # 然后，根据真实标签，获取模型对 "正确" 类别预测的概率 pt
        # .gather() 是一个方便的操作，用于根据索引（targets）从源张量（probs）中选取值
        pt = probs.gather(1, targets.unsqueeze(1))
        
        # --- 3. 计算最终的 Focal Loss ---
        # 核心公式: focal_loss = (1 - pt)^gamma * cross_entropy_loss
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss.unsqueeze(1)
        
        # 应用规约
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss