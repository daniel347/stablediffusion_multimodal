from torch import nn

class IdentityMeasurement(nn.Module):

    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        return x