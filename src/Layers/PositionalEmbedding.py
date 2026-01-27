from torch import nn

class Embedding(nn.Module):
    def __init__(self, dim_embedded, context_length):
        super().__init__()
