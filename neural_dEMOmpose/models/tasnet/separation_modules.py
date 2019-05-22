"""!
@brief Main separation module as in the proposed Tasnet model
@todo Convert it for classification purposes

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""

import torch.nn as nn
import torch.nn.functional as F


class DConvLayer(nn.Module):
    """
    1D dilated convolutional layers that perform D-Conv

    input shape: [batch, channels, window]
    output shape: [batch, channels, window]

    Args:
    out_channels: int, number of filters in the convolutional block
    kernel_size: int, length of the filter
    dilation: int, size of dilation
    """

    def __init__(self, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv = nn.Conv1d(out_channels,
                              out_channels,
                              kernel_size,
                              padding=dilation*(kernel_size-1)//2,
                              dilation=dilation, groups=out_channels)

    def forward(self, x):
        out = self.conv(x)
        return out

class SConvBlock(nn.Module):
    """
    Convolutional blocks that group S-Conv operations, including
    1x1 conv, prelu, normalization and D-Conv with dilation sizes

    input shape: [batch, in_channels, win]
    output shape: [batch, in_channels, win]

    Args:
    in_channels: int
    out_channels: int
    kernel_size: int (in paper, always set to 3)
    depth: int, log_2(dilation) (0-7 in the paper inclusive)
    """

    def __init__(self, in_channels, out_channels, kernel_size, depth):
        super().__init__()
        self.in_conv = nn.Conv1d(in_channels, out_channels, 1)
        self.dconv = DConvLayer(out_channels, kernel_size, 2**depth)
        self.out_conv = nn.Conv1d(out_channels, in_channels, 1)
        self.in_prelu = nn.PReLU(out_channels)
        self.in_bn = nn.BatchNorm1d(out_channels)
        self.out_prelu = nn.PReLU(out_channels)
        self.out_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        inp = x
        x = self.in_conv(x)
        x = self.in_prelu(x)
        x = self.in_bn(x)
        x = self.dconv(x)
        x = self.out_prelu(x)
        x = self.out_bn(x)
        x = self.out_conv(x)
        return inp + x

class SeparationModule(nn.Module):
    """
    Separation module in TasNet-like architecture. Applies masks of
    different classes on the encoded representation of the signal.

    input shape: [batch, N, win]
    output shape: [batch, N, win, C]

    Args:
        N: int, number of out channels in the encoder
        B: int, number of out channels in the bottleneck layer
        H: int, number of out channels in D-conv layer
        P: int, size of D-conv filter
        X: int, number of conv blocks in each repeat (max depth of dilation)
        R: int, number of repeats
    """

    def __init__(self, N=256, B=256, H=512, P=3, X=8, R=4, C=2):
        super().__init__()
        self.C = C
        self.bottleneck_conv = nn.Conv1d(N, B, 1)
        self.blocks = nn.ModuleList()
        for r in range(R):
            for x in range(X):
                self.blocks.append(SConvBlock(B, H, P, x))
        self.out_conv = nn.Conv1d(N, N*C, 1)

    def forward(self, x):
        m = self.bottleneck_conv(x)
        for i in range(len(self.blocks)):
            m = self.blocks[i](m)
        
        m = self.out_conv(m)
        m = m.unsqueeze(1).contiguous().view(m.shape[0],
                                             self.C,
                                             -1,
                                             m.shape[-1])
        m = F.softmax(m, dim=1)
        x = x.unsqueeze(1)
        m_out = x * m
        return m_out

