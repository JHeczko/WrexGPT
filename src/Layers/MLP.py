from torch import nn
import torch


class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()

        self.l1 = nn.Linear(dim_in, dim_hidden)

        self.gelu = nn.GELU()

        self.l2 = nn.Linear(dim_hidden, dim_in)
        self.l2.RESIDUAL_INIT = 1

    def forward(self, x):
        x = self.l1(x)
        x = self.gelu(x)
        x = self.l2(x)
        return x


if __name__ == '__main__':
    torch.manual_seed(0)

    batch_size = 1
    seq_len = 4
    dim = 4

    x = torch.randn(batch_size, seq_len, dim)

    print(x)

    mlp = MultiLayerPerceptron(dim, dim*4)
    layer_norm = nn.LayerNorm(dim)

    x_out = layer_norm(mlp(x))

    print(x_out)
    print(x_out.mean(dim=-1))
    print(x_out.var(dim=-1, unbiased=False))