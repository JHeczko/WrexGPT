import torch
from torch import nn
from Attention import MaskedMultiHeadAttention
from MLP import MultiLayerPerceptron

# Decoder only, transormer like block
class Transformer(nn.Module):
    def __init__(self, dim_embedded, context_length, num_heads=12, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.dim_embedded = dim_embedded
        self.context_length = context_length

        self.attention = MaskedMultiHeadAttention(dim_in=self.dim_embedded, dim_out=self.dim_embedded, context_length=context_length, num_heads=num_heads, dropout=dropout, bias=qkv_bias)

        self.mlp = MultiLayerPerceptron(dim_in=self.dim_embedded, dim_hidden=self.dim_embedded*4)

        self.ln1 = nn.LayerNorm(self.dim_embedded, elementwise_affine=False)
        self.ln2 = nn.LayerNorm(self.dim_embedded, elementwise_affine=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = x + self.dropout(self.attention(self.ln1(x)))
        x = x + self.dropout(self.mlp(self.ln2(x)))
        return x


if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 1
    context_len = 16
    dim = 768

    x = torch.randn(batch_size, context_len, dim)
    print(x.shape)

    transformer = Transformer(dim_embedded=dim, context_length=context_len)

    x_out = transformer(x)

    print(x_out.shape)