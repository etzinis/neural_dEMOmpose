"""!
@brief Encoder layer definition as in the proposed Tasnet model

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""
import torch.nn as nn
import torch.nn.functional as F


class EncodingLayer(nn.Module):
    '''
        A 1D convolutional block that transforms signal in wave form into higher dimension

        input shape: [batch, 1, sample]
        output shape: [batch, enc_channels, sample/kernel_size]

        Args:
            enc_channels: int, number of output channels for the encoding convolution
            kernel_size: int, length of the encoding filter

    '''
    def __init__(self, enc_channels, kernel_size):
        super().__init__()
        self.conv = nn.Conv1d(1,
                              enc_channels,
                              kernel_size,
                              stride=kernel_size//2)
    
    def forward(self, x):
        w = F.relu(self.conv(x))
        return w

