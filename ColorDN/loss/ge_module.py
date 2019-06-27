from torch import nn


class GELayer(nn.Module):
    def __init__(self):
        super(GELayer, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _,_,h,w = x.size()
        y = nn.functional.adaptive_avg_pool2d(x,(h//4, w//4))
        y = nn.functional.interpolate(y,(h,w))
        y = self.sigmoid(y)
        return x * y

class L1_softmax_layer(nn.Module):
    def __init__(self):
        super(L1_softmax_layer, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(x)
        return x * y
