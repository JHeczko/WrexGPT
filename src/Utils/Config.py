from dataclasses import dataclass
from typing import Literal
import torch


@dataclass
class ModelConfig:
    """Base model configuration that can be initialized for different GPT-2 variants."""
    dim_embedded: int
    context_length: int
    num_heads: int
    layers: int
    dropout: float
    vocab_size: int = 50258
    padding_token: int = 50257

    @classmethod
    def from_preset(cls, preset: Literal["gpt2", "gpt2-mini"] = "gpt2"):
        """Create a model configuration from a preset.

        Args:
            preset: Either "gpt2" or "gpt2-mini"
        """
        configs = {
            "gpt2-mini": {
                "dim_embedded": 512,
                "context_length": 512,
                "num_heads": 4,
                "layers": 6,
                "dropout": 0.2,
            },
            "gpt2": {
                "dim_embedded": 768,
                "context_length": 1024,
                "num_heads": 12,
                "layers": 12,
                "dropout": 0.1,
            }
        }

        if preset not in configs:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(configs.keys())}")

        return cls(**configs[preset])


    def __repr__(self) -> str:
        """Pretty print configuration."""
        class_name = self.__class__.__name__
        fields = []
        max_key_length = max(len(key) for key in self.__dataclass_fields__.keys())

        for key, value in self.__dict__.items():
            padding = " " * (max_key_length - len(key))
            fields.append(f"  {key}{padding} = {value}")

        fields_str = "\n".join(fields)
        return f"{class_name}(\n{fields_str}\n)"

@dataclass
class TrainConfig:
    """Base training configuration that can be initialized for different GPT-2 variants."""
    training_steps: int
    max_lr: float
    batch_size: int
    weight_decay: float
    grad_clip: float
    device: str
    scale_factor: float
    info_decay: int
    warmup_steps: int
    early_stopper_patience: int
    padding_token: int = 50257
    use_amp: bool = None

    def __post_init__(self):
        """Set use_amp based on CUDA availability if not explicitly set."""
        if self.use_amp is None:
            self.use_amp = torch.cuda.is_available()

    @classmethod
    def from_preset(cls, preset: Literal["gpt2", "gpt2-mini"] = "gpt2"):
        """Create a training configuration from a preset.

        Args:
            preset: Either "gpt2" or "gpt2-mini"
        """
        configs = {
            "gpt2-mini": {
                "training_steps": 4749500, # eqiuvalent of 300 epochs during training
                "max_lr": 5e-4,
                "batch_size": 64,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "scale_factor": 2.0,
                "warmup_steps": 500,
                "early_stopper_patience": 10,
                "info_decay": 1000,
            },
            "gpt2": {
                "training_steps": 355500, # equivalent of 300 epochs during training
                "max_lr": 2.5e-4,
                "batch_size": 256,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
                "scale_factor": 2.0,
                "warmup_steps": 2000,
                "early_stopper_patience": 10,
                "info_decay": 30,
            }
        }

        if preset not in configs:
            raise ValueError(f"Unknown preset: {preset}. Choose from {list(configs.keys())}")

        device = "cuda" if torch.cuda.is_available() else "cpu"

        return cls(
            **configs[preset],
            device=device,
        )

    def __repr__(self) -> str:
        """Pretty print configuration."""
        class_name = self.__class__.__name__
        fields = []
        max_key_length = max(len(key) for key in self.__dataclass_fields__.keys())

        for key, value in self.__dict__.items():
            padding = " " * (max_key_length - len(key))
            fields.append(f"  {key}{padding} = {value}")

        fields_str = "\n".join(fields)
        return f"{class_name}(\n{fields_str}\n)"

# Przykłady użycia:
if __name__ == "__main__":
    model_config_mini = ModelConfig.from_preset("gpt2-mini")
    train_config_mini = TrainConfig.from_preset("gpt2-mini")

    model_config = ModelConfig.from_preset("gpt2")
    train_config = TrainConfig.from_preset("gpt2")

    custom_model_config = ModelConfig(
        dim_embedded=1024,
        context_length=2048,
        num_heads=16,
        layers=24,
        dropout=0.1,
    )

    model_config_modified = ModelConfig.from_preset("gpt2")
    model_config_modified.dropout = 0.3

    print(model_config_mini)
    print(train_config)