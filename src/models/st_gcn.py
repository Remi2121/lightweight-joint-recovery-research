"""
ST-GCN baseline (Yan, Xiong & Lin, AAAI 2018), implemented per the
architecture spec recorded in the project's literature review:

  9 ST-GCN units: channels 64,64,64 / 128,128,128 / 256,256,256
  Temporal kernel Gamma=9 on every unit
  Residual connection on each unit
  Dropout 0.5 after each unit
  Stride 2 at the 4th and 7th units (temporal downsampling)
  Learnable edge-importance mask M per layer
  Global average pooling -> FC -> class logits

This is the baseline backbone this research plugs missing-joint
recovery into (upstream of this model, on the input skeleton).
"""

import torch
import torch.nn as nn

from .graph import Graph


class SpatialGraphConv(nn.Module):
    """K-subset spatial graph convolution: f_out = sum_k (A_k (x) M_k) W_k x."""

    def __init__(self, in_channels, out_channels, A):
        super().__init__()
        self.register_buffer("A", A)  # (K, V, V)
        K = A.size(0)
        self.conv = nn.Conv2d(in_channels, out_channels * K, kernel_size=1)
        self.K = K
        self.out_channels = out_channels

    def forward(self, x, edge_importance):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        x = self.conv(x)  # (N, out*K, T, V)
        x = x.view(N, self.K, self.out_channels, T, V)
        A = self.A * edge_importance  # (K, V, V)
        x = torch.einsum("nkctv,kvw->nctw", x, A)
        return x.contiguous()


class STGCNUnit(nn.Module):
    def __init__(self, in_channels, out_channels, A, temporal_kernel=9, stride=1, residual=True, dropout=0.5):
        super().__init__()
        self.gcn = SpatialGraphConv(in_channels, out_channels, A)
        pad = (temporal_kernel - 1) // 2
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=(temporal_kernel, 1),
                      stride=(stride, 1), padding=(pad, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.bn_in = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x, edge_importance):
        res = self.residual(x)
        x = self.gcn(x, edge_importance)
        x = self.bn_in(x)
        x = self.tcn(x)
        x = x + res
        return self.relu(x)


class STGCN(nn.Module):
    """
    Input:  (N, C=3, T, V=25)  -- single-person skeleton clip
    Output: (N, num_classes)
    """

    CHANNEL_PLAN = [
        (3, 64, 1), (64, 64, 1), (64, 64, 1),
        (64, 128, 2), (128, 128, 1), (128, 128, 1),
        (128, 256, 2), (256, 256, 1), (256, 256, 1),
    ]  # (in, out, stride) for 9 units; strides of 2 at units 4 and 7

    def __init__(self, num_classes, num_joints=25, temporal_kernel=9, dropout=0.5):
        super().__init__()
        graph = Graph(num_joints=num_joints)
        A = torch.tensor(graph.A, dtype=torch.float32)
        self.register_buffer("A_template", A)

        self.data_bn = nn.BatchNorm1d(3 * num_joints)

        self.units = nn.ModuleList([
            STGCNUnit(in_c, out_c, A, temporal_kernel=temporal_kernel, stride=stride,
                      residual=(i != 0), dropout=dropout)
            for i, (in_c, out_c, stride) in enumerate(self.CHANNEL_PLAN)
        ])

        # one learnable edge-importance mask per unit (Yan et al. Sec 3.5)
        self.edge_importance = nn.ParameterList([
            nn.Parameter(torch.ones(A.size())) for _ in self.CHANNEL_PLAN
        ])

        self.fc = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.shape
        x = x.permute(0, 1, 3, 2).contiguous().view(N, C * V, T)
        x = self.data_bn(x)
        x = x.view(N, C, V, T).permute(0, 1, 3, 2).contiguous()

        for unit, importance in zip(self.units, self.edge_importance):
            x = unit(x, importance)

        x = nn.functional.adaptive_avg_pool2d(x, 1)  # (N, 256, 1, 1)
        x = self.fc(x)
        return x.view(x.size(0), -1)
