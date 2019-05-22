import time, sys, os
import torch
import torch.nn as nn
import torch.nn.functional as F
root_dir = os.path.join(
   os.path.dirname(os.path.realpath(__file__)),
   '../../../')
sys.path.insert(0, root_dir)
from end2end_unsupervised_separation.dnn.models.encoder import \
   EncodingLayer
from end2end_unsupervised_separation.dnn.models.decoder import \
   DecodingLayer
from end2end_unsupervised_separation.dnn.models.separation_modules \
   import SeparationModule
import pdb

class TasNet(nn.Module):
    '''
        TasNet Architecture
        
        input shape: [batch, 1, sample]
        output shape: [batch, C, sample]

        
        Args:
            N: int, number of out channels in the encoder 
            L: int, size of encoding filter (window length)
            B: int, number of out channels in the bottleneck layer
            H: int, number of out channels in D-conv layer 
            P: int, size of D-conv filter 
            X: int, number of conv blocks in each repeat (max depth of dilation)
            R: int, number of repeats

    '''
    def __init__(self, N=256, L=20, B=256, H=512, P=3, X=8, R=4, C=2):
        super().__init__()
        self.encoder = EncodingLayer(N, L)
        self.sep_module = SeparationModule(N, B, H, P, X, R, C)
        self.decoder = DecodingLayer(N, L)
        #  self.decoder = nn.ConvTranspose2d(C, C, (N, L), stride=(1, L//2), groups=C)

    def forward(self, x):
        x = self.encoder(x)
        m_out = self.sep_module(x)
        #  x_hat = self.decoder(m_out)
        x_hat = torch.cat([self.decoder(m_out[:, c, :, :]) for c in range(m_out.shape[1])], dim=1)
        return x_hat


if __name__ == '__main__':
    N = 256
    L = 20
    B = 256
    H = 512
    P = 3
    X = 8
    R = 4
    C = 2

    device = 'cuda:2'
    sig = torch.randn(4, 1, 32000).to(device)
    model = TasNet(N, L, B, H, P, X, R, C).to(device)
    print('==> Number of trainable params: {}'.format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    t0 = time.time()
    out = model(sig)
    print(time.time()-t0)
    pdb.set_trace()

