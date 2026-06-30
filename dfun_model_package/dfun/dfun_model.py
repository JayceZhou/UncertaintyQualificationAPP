from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    """DFUN / gated PhyNetCNN model used for 230-class space-group inference.

    Inputs:
        x_xrd: Tensor with shape [batch_size, raw_xrd_length].
        x_phys: Tensor with shape [batch_size, 45].

    Output:
        logits: Tensor with shape [batch_size, 230].

    Gate definition:
        gated_xrd_features = g * xrd_features
        gated_phys_features = (1 - g) * phys_features

    Therefore, a larger gate value means stronger reliance on the CNN/raw-XRD
    branch, while a smaller gate value means stronger reliance on the
    peak/physical-feature branch.
    """

    def __init__(self, raw_xrd_length: int = 3501, physical_features_length: int = 45, num_classes: int = 230):
        super().__init__()
        self.raw_xrd_length = raw_xrd_length
        self.physical_features_length = physical_features_length
        self.num_classes = num_classes

        self.cnn_branch = nn.Sequential(
            nn.Conv1d(1, 40, 100, 5),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(40, 80, 50, 5),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Conv1d(80, 80, 25, 2),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.Dropout(0.4),
        )

        self.gate_xrd_projector = nn.Linear(12160, 128)

        self.mlp_branch = nn.Sequential(
            nn.Linear(physical_features_length, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
        )

        self.gating_network = nn.Sequential(
            nn.Linear(12160 + 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        self.classifier_head = nn.Sequential(
            nn.Linear(12160 + 64, 2300),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(2300, 1150),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1150, num_classes),
        )

    def forward(self, x_xrd: torch.Tensor, x_phys: torch.Tensor, return_gate: bool = False):
        x_xrd = x_xrd.unsqueeze(1)
        x_xrd = F.interpolate(x_xrd, size=8500, mode="linear", align_corners=False)
        xrd_features = self.cnn_branch(x_xrd)
        xrd_features = xrd_features.reshape(xrd_features.shape[0], -1)

        phys_features = self.mlp_branch(x_phys)

        gate_input = torch.cat((xrd_features, phys_features), dim=1)
        gate = self.gating_network(gate_input)

        gated_xrd_features = gate * xrd_features
        gated_phys_features = (1.0 - gate) * phys_features

        merged = torch.cat((gated_xrd_features, gated_phys_features), dim=1)
        logits = self.classifier_head(merged)

        if return_gate:
            return logits, gate
        return logits


def load_dfun_checkpoint(checkpoint_path: str, device: str | torch.device = "cpu") -> Model:
    """Load the packaged DFUN checkpoint."""
    device = torch.device(device)
    model = Model().to(device)
    try:
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def enable_mc_dropout(model: nn.Module) -> List[str]:
    """Enable MC Dropout while keeping BatchNorm and all other modules in eval mode.

    Usage:
        model.eval()
        enabled = enable_mc_dropout(model)

    The returned list contains the names of Dropout modules switched to train
    mode. No parameters are updated during inference if the caller uses
    torch.no_grad().
    """
    enabled: List[str] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            module.train()
            enabled.append(name)
    return enabled


def count_dropout_modules(model: nn.Module) -> List[Tuple[str, float]]:
    """Return dropout module names and probabilities."""
    modules: List[Tuple[str, float]] = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            modules.append((name, float(module.p)))
    return modules
