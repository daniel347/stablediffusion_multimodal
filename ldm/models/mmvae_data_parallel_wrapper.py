from torch import nn


class MMVAEDataParallelWrapper(nn.Module):

    def __init__(self, model):
        super(MMVAEDataParallelWrapper, self).__init__()
        self.module = model

    def forward(self, function, *args, **kwargs):
        return getattr(self.module, function)(*args, **kwargs)