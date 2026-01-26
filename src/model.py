from torch import nn

# GPT-2 Pretrained only
class WrexGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config