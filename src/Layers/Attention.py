from torch import nn
import torch
import math

class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length, num_heads=12, dropout=0.1, bias=False):
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

        self.softmax = nn.Softmax(dim=-1)

        self.register_buffer("mask", torch.triu(torch.ones(context_length, context_length), diagonal=1))

        self.out_projection = nn.Linear(dim_out, dim_out)
        self.out_projection.RESIDUAL_INIT = 1

    # INPUT x = (batch_size = 2, context_len = 3, dim_in = 968)
    def forward(self, x, padding_mask=None):
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
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # now we want to have (batch_size, num_heads, context_len, context_len)
        # x = (batch_size = 2, num_heads = 8, context_len = 3, context_len = 3)
        att_score = q @ k.transpose(2, 3)
        att_score = att_score / math.sqrt(self.head_dim)

        # now masking
        # but if sentences are not == context_len then mask is truncated
        mask = self.mask.bool()[0:context_length, 0:context_length]
        # trio mask
        att_score = att_score.masked_fill_(mask, torch.finfo(att_score.dtype).min)
        # padding mask
        if padding_mask is not None:
            att_score = att_score.masked_fill_(padding_mask, torch.finfo(att_score.dtype).min)



        # softmax + dropout
        att_score = self.softmax(att_score)
        att_score = self.dropout(att_score)

        # (batch_size, num_heads, context_len, context_len) * (batch_size, num_heads,context_len, head_dim)
        # we want to have once again
        # (batch_size = 2, num_heads = 8,context_len = 3, head_dim = 121) not attention scores
        context_vec = att_score @ v

        # now lets revert to
        # (batch_size = 2,context_len = 3, num_heads = 8, head_dim = 121)
        context_vec = context_vec.transpose(1, 2)

        # now lets concatenate all the head so we have like at the beggining
        # (batch_size = 2, context_len = 3, dim_out = 968)
        context_vec = context_vec.contiguous().view(batch_size, context_length, self.dim_out)
        context_vec = self.out_projection(context_vec)
        return context_vec




if __name__ == "__main__":
    torch.manual_seed(0)

    batch_size = 2
    seq_len = 4
    dim = 4
    heads = 2

    x = torch.randn(batch_size, seq_len, dim)

    print(x.shape)

    my_attn = MaskedMultiHeadAttention(
        dim_in=dim,
        dim_out=dim,
        context_length=seq_len,
        num_heads=heads,
        dropout=0.0,
        bias=False,
    )

    my_attn.eval()

    my_out = my_attn(x)

    print(my_out.shape)
