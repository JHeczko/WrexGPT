from dataclasses import dataclass
import torch

@dataclass
class ConfigGPT2:
    dim_embedded: int = 768
    vocab_size: int = 50258
    context_length: int = 1024
    num_heads: int = 12
    layers: int = 12
    padding_token: int = 50257  # Możesz od razu ustawić domyślną wartość

class TrainConfig:
    epochs: int = 5
    lr: float = 3e-4
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # mixed precision
    use_amp: bool = True

    # logging
    log_every: int = 50

    # optional: scheduler
    warmup_steps: int = 0