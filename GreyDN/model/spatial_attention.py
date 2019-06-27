import torch
from torch import nn

class SpatialAttn(nn.Module):
    """Spatial Attention Layer"""
    def __init__(self):
        super(SpatialAttn, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # global cross-channel averaging # e.g. 32,2048,24,8
        xx = x
        x = x.mean(1, keepdim=True)  # e.g. 32,1,24,8
        n,_,h,w = x.size()
        for b in range(n):
            x[b] /= self.sigmoid(x[b])
        return xx*x
