# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)

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
        out = self.conv(x)
        return torch.cat((x, out), 1)

class RDB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(RDB, self).__init__()
        G0 = growRate0
        G  = growRate
        C  = nConvLayers
        
        convs = []
        for c in range(C):
            convs.append(RDB_Conv(G0 + c*G, G))
        self.convs = nn.Sequential(*convs)
        
        # Local Feature Fusion
        self.LFF = nn.Conv2d(G0 + C*G, G0, 1, padding=0, stride=1)

    def forward(self, x):
        return self.LFF(self.convs(x))  + x

class RDN(nn.Module):
    def __init__(self, args):
        super(RDN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.RDNkSize

        # number of RDB blocks, conv layers, out channels
        self.D, C, G = {
            'A': (20, 6, 32),
            'B': (16, 8, 64),
        }[args.RDNconfig]

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2_b1 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2_b2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        #multi-branch
        self.branch1_1 = RDB(growRate0 = G0, growRate = G, nConvLayers = C)
        self.branch1_2 = RDB(growRate0 = G0, growRate = G, nConvLayers = C)
        self.branch1_3 = RDB(growRate0 = G0, growRate = G, nConvLayers = C)
        self.GFF_b1 = nn.Sequential(*[
            nn.Conv2d(3 * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        self.branch2_1 = RDB(growRate0 = G0, growRate = G, nConvLayers = C)
        self.branch2_2 = RDB(growRate0 = G0, growRate = G, nConvLayers = C)
        self.branch2_3 = RDB(growRate0 = G0, growRate = G, nConvLayers = C)
        self.GFF_b2 = nn.Sequential(*[
            nn.Conv2d(3 * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            nn.Conv2d(2 * G0, G0, 1, padding=0, stride=1),
            nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)
        ])

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
        x_1 = self.SFENet2_b1(f__1)
        x_2 = self.SFENet2_b2(f__1)

        branch1_1 = self.branch1_1(x_1)
        branch1_2 = self.branch1_2(x_1)
        branch1_3 = self.branch1_3(x_1)
        b1 = self.GFF_b1(torch.cat([branch1_1, branch1_2, branch1_3], 1))
        branch2_1 = self.branch2_1(x_2)
        branch2_2 = self.branch2_2(x_2)
        branch2_3 = self.branch2_3(x_2)
        b2 = self.GFF_b2(torch.cat([branch2_1, branch2_2, branch2_3], 1))

        x = self.GFF(torch.cat([b1, b2],1))
        x += f__1

        return self.UPNet(x)
