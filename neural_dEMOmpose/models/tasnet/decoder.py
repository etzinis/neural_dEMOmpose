"""!
@brief Decoder layer definition as in the proposed Tasnet model

@author Efthymios Tzinis {etzinis2@illinois.edu}
@copyright University of Illinois at Urbana Champaign
"""
import torch.nn as nn


class DecodingLayer(nn.Module):
    '''
        A 1D deconvolutional block that transforms encoded representation into wave form

        input shape: [batch, enc_channels, window]
        output shape: [batch, 1, window*kernel_size]


        Args:
            enc_channels: number of output channels for the encoding convolution
            kernel_size: length of the encoding filter

    '''
    def __init__(self, enc_channels, kernel_size):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(enc_channels,
                                         1,
                                         kernel_size,
                                         stride=kernel_size//2)

    def forward(self, x):
        return self.deconv(x)

