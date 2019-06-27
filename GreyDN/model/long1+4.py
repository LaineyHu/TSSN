# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn
from .se_module import SELayer

import cv2
import numpy as np
from scipy import misc

def feature_norm(x, eps=1e-8):
    return (x-x.min()) / (x.max()-x.min()+eps)

def make_model(args, parent=False):
    return RDN(args)

def expFetch(lst):
    nums = len(lst)
    i = 1
    res = []
    while i <= nums:
        res.append(lst[nums - i])
        i *= 2
    return res

class RDB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(RDB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        return self.conv(x)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        self.C  = nConvLayers
        self.prevChannels = [G0]
        self.expFetch = expFetch
        
        for i in range(self.C):
            nChannels = sum(self.expFetch(self.prevChannels))
            unit = RDB_Conv(nChannels, G)
            self.add_module("block-%d" % (i+1), unit)
            self.prevChannels.append(G)
        
        self.nOutChannels = sum(self.expFetch(self.prevChannels))
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(self.nOutChannels, G0, 1, padding=0, stride=1)

    def forward(self, x):
        prev_outputs = [x]
        x_tmp = x
        for i in range(self.C):
            out = self._modules["block-%d" %(i+1)](x)
            prev_outputs.append(out)
            fetch_outputs = self.expFetch(prev_outputs)
            x = torch.cat(fetch_outputs, 1).contiguous() 

        return self.LFF(x) + x_tmp

class TWO(nn.Module):
    def __init__(self,args):
        super(TWO, self).__init__()
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 16, 64),
        }[args.RDNconfig]


        self.branch1 = RDB(growRate0 = G0, growRate = G, nConvLayers = C)
        self.branch2 = nn.Sequential(*[
            RDB(growRate0 = G0, growRate = G, nConvLayers = C),
            RDB(growRate0 = G0, growRate = G, nConvLayers = C),
            RDB(growRate0 = G0, growRate = G, nConvLayers = C),
            RDB(growRate0 = G0, growRate = G, nConvLayers = C)
        ])
        self.fusion = nn.Sequential(*[
            SELayer(2*G0),
            nn.Conv2d(2*G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        fusion = self.fusion(torch.cat([x1,x2],1))
        return x+fusion
        
class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 16, 64),
        }[args.RDNconfig]

        self.B = 4
        
        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        blocks = [TWO(args) for _ in range(self.B)]
        self.block = nn.Sequential(*blocks)
        
        # Global Feature Fusion
        self.GFF = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # Up-sampling net
        if r == 2 or r == 3:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * r * r, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(r),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        elif r == 4:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])
        else:
            raise ValueError("scale must be 2 or 3 or 4.")

    def forward(self, x):
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)
        
        x = self.block(x)

        x = self.GFF(x)

        x += f__1

        return self.UPNet(x)
