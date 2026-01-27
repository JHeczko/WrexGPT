from torch import nn
import Utils
import Layers
import torch

# GPT-2 Pretrained only
class WrexGPT(nn.Module):
    def __init__(self, config: Utils.ConfigGPT):
        super().__init__()
        self.config = config


        self.embedding = nn.Embedding(config.vocab_size, config.dim_embedded)
        self.positional_encoding = Layers.PositionalEncoding(config.context_length, config.dim_embedded)

        self.transformers = nn.ModuleList()
        for _ in range(config.layers):
            self.transformers.append(
                Layers.TransformerDecoder(
                    dim_embedded=config.dim_embedded,
                    context_length=config.context_length,
                    num_heads=config.num_heads,
                    dropout=0.1,
                    qkv_bias=False
                )
            )

        self.out_ln = nn.LayerNorm(config.dim_embedded)
        self.out_projection = nn.Linear(config.dim_embedded, config.vocab_size)

    # x = (batch_size, >= context_len)
    def forward(self, x):
        batch_size, sentence_length = x.shape
        assert sentence_length <= self.config.context_length

        # first we embed
        # x = (batch_size, context_len, embedded_dim)
        emb_x = self.embedding(x)

        # then add postional learned encodding
        # x = (batch_size, context_len, embedded_dim)
        pos_x = self.positional_encoding(sentence_length, x.device)

        # summing up
        x = emb_x + pos_x

        # now grind through transformers

        for transformer in self.transformers:
            x = transformer(x)

        # finally normalization and linear layer
        # x = (batch_size, context_len, vocab_size)
        x = self.out_ln(x)
        x = self.out_projection(x)

        return x




if __name__ == "__main__":
    config = Utils.ConfigGPT(
        dim_embedded=12,
        vocab_size=500,
        context_length=12,
        num_heads=2,
        layers=2
    )
    model = WrexGPT(config)

    dummy_vec = torch.ones(10, config.context_length, dtype=torch.int)

    out_vec = model(dummy_vec)

    print(out_vec.shape)