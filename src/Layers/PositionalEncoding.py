from torch import nn
import torch

class PositionalEncoding(nn.Module):
    def __init__(self, context_length, dim_embedded):
        super().__init__()
        self.position_embedding = nn.Embedding(context_length, dim_embedded)

    def forward(self, sentence_length, device):
        # pos = (, context_len)
        pos = torch.arange(sentence_length, device=device, dtype=torch.long)

        # pos_emb = (, context_len, embedded_dim)
        pos_emb = self.position_embedding(pos)

        return pos_emb




if __name__ == '__main__':
    vocab_size = 51212
    context_length = 1024
    batch_size = 4
    embd_dim = 512

    embedding = nn.Embedding(vocab_size, embd_dim)

    dummy_vector = torch.ones(batch_size, context_length, dtype=torch.int)

    print(dummy_vector.shape)

    out_embedding = embedding(dummy_vector)

    # torch.Size([4, 51212, 512])
    print(out_embedding.shape)
