from torch import nn
import torch

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, num_heads=8, dropout=0.1, bias=False):
        super().__init__()

        self.dim_in = dim_in
        self.dim_out = dim_out

        if (dim_out % num_heads) != 0:
            raise ValueError("dim_out must be divisible by num_heads")

        self.num_heads = num_heads
        self.context_length = context_length
        self.dropout = nn.Dropout(dropout)
        self.head_dim = dim_out // num_heads

        self.w_query = nn.Linear(dim_in, dim_out, bias=bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=bias)

        self.softmax = nn.Softmax()

        self.re

    # INPUT x = (batch_size = 2, context_len = 3, dim_in = 968)
    def forward(self, x):
        batch_size, context_length, dim_in = x.shape

        # x = (batch_size = 2, context_len = 3, dim_out = 968)
        q = self.w_query(x)
        k = self.w_key(x)
        v = self.w_value(x)

        # we have to split each encoded sentence for head
        # x = (batch_size = 2, context_len = 3, num_heads = 8, head_dim = 121)
        q = q.view(batch_size,context_length, self.num_heads, self.head_dim)
        k = k.view(batch_size,context_length, self.num_heads, self.head_dim)
        v = v.view(batch_size,context_length, self.num_heads, self.head_dim)

        # now we have per batch-head sample, each batch, have heads wich have specific subspace of mapping
        # x = (batch_size = 2, num_heads = 8,context_len = 3, head_dim = 121)
        q = q.transpose(1, 2)
        w = k.transpose(1, 2)
        k = k.transpose(1, 2)

        # now we want to have (batch_size, num_heads, context_len, context_len)
        att_score = q @ k.transpose(2, 3)
        # now masking
        att_score =



if __name__ == "__main__":
    torch.manual_seed(123)

    inputs = torch.tensor(
    [[0.51, 0.23, 0.76, 0.48, 0.92, 0.61],  # Token 1 representation
     [0.65, 0.79, 0.58, 0.31, 0.67, 0.42],  # Token 2 representation
     [0.84, 0.19, 0.14, 0.09, 0.77, 0.50]]  # Token 3 representation
    )

    # (batch_size = 2, context_len = 3, d_in = 6)
    batch = torch.stack((inputs, inputs), dim=0)

    print(batch.shape)

