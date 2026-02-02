from torch import nn
import Utils
import Layers
import torch

# GPT-2 Pretrained only
class WrexGPT(nn.Module):
    def __init__(self, config: Utils.ConfigGPT2):
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

        self.apply(self.__init_weights)

    def __init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            std = 0.02

            if hasattr(module, 'RESIDUAL_INIT'):
                std /= (2*self.config.layers)**0.5

            nn.init.normal_(module.weight, mean=0,std=std)

            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0, std=0.02)

    # x = (batch_size, >= context_len)
    def forward(self, x):
        batch_size, sentence_length = x.shape
        assert sentence_length <= self.config.context_length

        # before all compute we have to calculate padding mask
        padding_token = self.config.padding_token

        # keep = (batch_size, context_len)
        keep = (x != padding_token)  # (B, C) bool, True = token
        # keep2d = (batch_size, context_len, context_en)
        keep2d = keep.unsqueeze(2) & keep.unsqueeze(1)
        mask_pad = ~keep2d
        # mask_pad = (batch_size, 1, context_len, context_len)
        mask_pad = mask_pad.unsqueeze(1)

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
            x = transformer(x, mask_pad)

        # finally normalization and linear layer
        # x = (batch_size, context_len, vocab_size)
        x = self.out_ln(x)
        x = self.out_projection(x)

        # logits out
        return x


if __name__ == "__main__":
    config = Utils.ConfigGPT2(
        dim_embedded=12,
        vocab_size=500,
        context_length=12,
        num_heads=2,
        layers=2,
        padding_token=50257
    )
    model = WrexGPT(config)

    dummy_vec = torch.ones(10, config.context_length-4, dtype=torch.int)

    out_vec = model(dummy_vec)

    print(out_vec.shape)