from torch import nn


class MMVAEDataParallelWrapper(nn.Module):

    def __init__(self, model):
        super(MMVAEDataParallelWrapper, self).__init__()
        self.module = model

    def forward(self, function, *args, **kwargs):
        """As torch data distributed parallel by default only splits inputs to the forward function, we wrap the model and call the function by string name."""
        return getattr(self.module, function)(*args, **kwargs)