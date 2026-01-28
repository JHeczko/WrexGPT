import torch
import torch.nn.functional as F


class AutoregressiveGenerator:
    def __init__(self, model, context_length: int, device: str = None):
        """
        model: Twój GPT, forward(idx) -> logits (B, T, vocab_size)
        context_length: maksymalny kontekst modelu
        """
        self.model = model
        self.context_length = context_length
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        greedy: bool = False,
    ) -> torch.Tensor:
        """
        idx: (T,) albo (B, T) - tokeny startowe
        return: (B, T + max_new_tokens)
        """
        self.model.eval()

        if idx.dim() == 1:
            idx = idx.unsqueeze(0)  # (1, T)

        idx = idx.to(self.device)

        for _ in range(max_new_tokens):
            # ucinamy kontekst (żeby nie przekroczyć context_length)
            idx_cond = idx[:, -self.context_length:]  # (B, T_ctx)

            logits = self.model(idx_cond)             # (B, T_ctx, V)
            logits = logits[:, -1, :]                 # (B, V) tylko ostatni token

            # temperature
            logits = logits / max(temperature, 1e-8)

            # top-k sampling
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")

            probs = F.softmax(logits, dim=-1)  # (B, V)

            # wybór następnego tokenu
            if greedy:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)  # (B,1)
            else:
                next_token = torch.multinomial(probs, num_samples=1)    # (B,1)

            idx = torch.cat([idx, next_token], dim=1)

        return idx
