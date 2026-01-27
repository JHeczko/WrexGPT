from torch import nn
from Utils import ConfigGPT

# GPT-2 Pretrained only
class WrexGPT(nn.Module):
    def __init__(self, config: ConfigGPT):
        super().__init__()
        self.config = config

        #
        self.embedding = nn.Embedding(config.vocab_size, config.dim_embedded)