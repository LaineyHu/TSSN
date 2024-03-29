import torch
import torch.nn as nn
from model import common

from scipy import misc

def feature_norm(x, eps=1e-8):
    return (x-x.min()) / (x.max()-x.min()+eps)

def make_model(args, parent=False):
    return TSSN(args)

def expFetch(lst):
    nums = len(lst)
    i = 1
    res = []
    while i <= nums:
        res.append(lst[nums - i])
        i *= 2
    return res

class SRB_Conv(nn.Module):
    def __init__(self, inChannels, growRate, kSize=3):
        super(SRB_Conv, self).__init__()
        Cin = inChannels
        G  = growRate
        self.conv = nn.Sequential(*[
            nn.Conv2d(Cin, G, kSize, padding=(kSize-1)//2, stride=1),
            nn.ReLU()
        ])

    def forward(self, x):
        return self.conv(x)

class SRB(nn.Module):
    def __init__(self, growRate0, growRate, nConvLayers, kSize=3):
        super(SRB, self).__init__()
        G0 = growRate0
        G  = growRate
        self.C  = nConvLayers
        self.prevChannels = [G0]
        self.expFetch = expFetch
        
        for i in range(self.C):
            nChannels = sum(self.expFetch(self.prevChannels))
            unit = SRB_Conv(nChannels, G)
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

class TSSN(nn.Module):
    def __init__(self, args):
        super(TSSN, self).__init__()
        r = args.scale[0]
        G0 = args.G0
        kSize = args.TSSNkSize
        
        rgb_mean = (0.4488, 0.4371, 0.4040)
        rgb_std = (1.0, 1.0, 1.0)
        self.sub_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std)

        #conv layers, out channels
        C = 16 
        G = 64

        # Shallow feature extraction net
        self.SFENet1 = nn.Conv2d(args.n_colors, G0, kSize, padding=(kSize-1)//2, stride=1)
        self.SFENet2 = nn.Conv2d(G0, G0, kSize, padding=(kSize-1)//2, stride=1)

        # multi-branch
        self.branch1 = nn.Sequential(*[
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C)
        ])
        self.branch2 = nn.Sequential(*[
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C),
            SRB(growRate0 = G0, growRate = G, nConvLayers = C)
        ])

        # Global Feature Fusion
        self.GFF = nn.Sequential(*[
            common.SELayer(2*G0),
            nn.Conv2d(2*G0, G0, kSize, padding=(kSize-1)//2, stride=1)
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
        elif r == 8:
            self.UPNet = nn.Sequential(*[
                nn.Conv2d(G0, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, G * 4, kSize, padding=(kSize-1)//2, stride=1),
                nn.PixelShuffle(2),
                nn.Conv2d(G, args.n_colors, kSize, padding=(kSize-1)//2, stride=1)
            ])

        else:
            raise ValueError("scale must be 2 or 3 or 4.")

        self.add_mean = common.MeanShift(args.rgb_range, rgb_mean, rgb_std, 1)

    def forward(self, x):
        x = self.sub_mean(x)
        f__1 = self.SFENet1(x)
        x  = self.SFENet2(f__1)

        branch1 = self.branch1(x)
        #misc.imsave("tmp/branch1.png", feature_norm(torch.squeeze(torch.mean(branch1*branch1, dim=1)).cpu().numpy()))
        branch2 = self.branch2(x)
        #misc.imsave("tmp/branch2.png", feature_norm(torch.squeeze(torch.mean(branch2*branch2, dim=1)).cpu().numpy()))

        x = self.GFF(torch.cat([branch1,branch2],1))
        x += f__1
        x = self.UPNet(x) 

        return self.add_mean(x)
