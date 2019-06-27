from torch import nn

class L1_softmax_layer(nn.Module):
    def __init__(self):
        super(L1_softmax_layer, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.sigmoid(x)
        return x * y
