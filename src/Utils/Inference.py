import torch
import torch.nn.functional as F
from torch import nn

import numpy as np

from src.Data import Tokenizer
from .Config import ModelConfig


class AutoregressiveGenerator:  # Nie musi dziedziczyć po nn.Module, jeśli nie ma trenowalnych parametrów
    def __init__(self, model, config, device):
        self.model = model
        self.config = config
        self.tokenizer = Tokenizer()  # Przekazujemy instancję, a nie tworzymy nową
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 50, temperature=1.0, top_k=None, greedy=True):
        tokens = torch.tensor(self.tokenizer.encode(text), dtype=torch.long, device=self.device)
        # added bacth
        tokens = tokens.view(1,-1)

        for _ in range(max_new_tokens):
            cond_tokens = None
            if tokens.shape[1] <= self.config.context_length:
                cond_tokens = tokens
            else:
                cond_tokens = tokens[:,-self.config.context_length:]
            logits = self.model(cond_tokens)
            logits = logits[:, -1, :]
            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            if greedy:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)

            tokens = torch.cat((tokens, next_token), dim=1)

        return self.tokenizer.decode(tokens[0].tolist())
