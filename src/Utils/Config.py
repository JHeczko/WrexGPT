from dataclasses import dataclass
import torch

@dataclass
class ConfigGPT2:
    dim_embedded: int = 768
    vocab_size: int = 50258
    context_length: int = 1024
    num_heads: int = 12
    layers: int = 12
    padding_token: int = 50257

class TrainConfig:
    epochs: int = 5
    max_lr: float = 2.5e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # mixed precision
    use_amp: bool =True
    scale_factor: float = 2.0

    # optional: scheduler
    warmup_steps: int = 2000

    padding_token: int = 50257
