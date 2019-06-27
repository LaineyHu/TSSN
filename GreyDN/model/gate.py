# Residual Dense Network for Image Super-Resolution
# https://arxiv.org/abs/1802.08797

from model import common

import torch
import torch.nn as nn


def make_model(args, parent=False):
    return RDN(args)

class GateB(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(GateB, self).__init__()
        Cin = inChannels
        G = growRate
        self.hidden = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU(),
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1)
        ])
        self.input = SELayer(2*Cin)
        self.forget = nn.Sequential(*[
            nn.Conv2d(2*Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            GELayer_new()
        ])
        self.update = nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1)
        self.output = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            GELayer_new()
        ])

    def forward(self, x, h):
        c = self.hidden(x)
        x_h = self.input(torch.cat([x,h],1))
        gate_forget = self.forget(x_h)
        gate_update = self.update(x_h)
        gate_output = self.output(x_h)
        forgeted = c.mul(gate_forget)
        updated = forgeted + gate_update.mul(1-gate_forget)
        output = updated.mul(gate_output)
        return updated, output
        

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

        # Redidual dense blocks and dense feature fusion
        blocks = [GateB(growRate0 = G0, growRate = G) for _ in range(self.D)]
        self.RDBs = nn.Sequential(*blocks)
        
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
        h = f__1
        
        xx, hh = self.RDBs(x, h)

        x = self.GFF(xx)
        x += f__1

        return self.UPNet(x)
