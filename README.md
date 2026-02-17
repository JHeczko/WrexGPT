# WrexGPT - GPT-2 Implementation

A complete PyTorch implementation of the GPT-2 language model from scratch, featuring a decoder-only transformer architecture, multi-head attention mechanisms, and advanced training utilities.

---

## Table of Contents

1. [Description](#description)
2. [Project Specification](#project-specification)
   - [Libraries](#libraries)
   - [Architecture](#architecture)
   - [Dataset](#dataset)
   - [Training](#training)
3. [Core Components](#core-components)
4. [Usage](#usage)
5. [References](#references)

---

## Description

**WrexGPT** is a clean, educational implementation of the GPT-2 architecture built from the ground up with PyTorch. This project demonstrates how modern large language models work by implementing every component—from tokenization to inference—with detailed explanations and modular design.

The model is capable of:
- **Text Generation**: Autoregressive generation with multiple decoding strategies (greedy, top-k sampling)
- **Language Modeling**: Next-token prediction trained on arbitrary text corpora
- **Configurable Sizes**: Support for multiple model variants (mini, standard, full)
- **Advanced Training**: Gradient accumulation, mixed precision training, early stopping, and checkpoint management

---

## Project Specification

### Libraries

| Library | Version | Purpose |
|---------|---------|---------|
| **PyTorch** | Latest | Deep learning framework, neural network layers |
| **NumPy** | Latest | Efficient array operations for data handling |
| **tiktoken** | Latest | GPT-2 tokenization (BPE encoding) |
| **ftfy** | Latest | Text cleaning and unicode normalization |
| **matplotlib** | Latest | Training history visualization |
| **tqdm** | Latest | Progress bars for training loops |

### Architecture

**WrexGPT** uses a **decoder-only Transformer** architecture, similar to GPT-2. Here's the architecture overview:

```
┌─────────────────────────┐
│   Input Tokens (B, L)   │
└────────────┬────────────┘
             │
             ▼
    ┌────────────────────┐
    │  Token Embedding   │  (B, L, D)
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │ Positional Encoding│  (B, L, D)
    └─────────┬──────────┘
              │
              ▼
    ┌─────────────────────────────────────┐
    │  Transformer Decoder Stack (N×)     │
    │  ┌──────────────────────────────┐   │
    │  │ Layer Norm                   │   │
    │  │ ▼                            │   │
    │  │ Multi-Head Attention         │   │
    │  │ Residual Connection + Dropout│   │
    │  │ ▼                            │   │
    │  │ Layer Norm                   │   │
    │  │ ▼                            │   │
    │  │ Feed-Forward Network (MLP)   │   │
    │  │ Residual Connection + Dropout│   │
    │  └──────────────────────────────┘   │
    └──────────────┬──────────────────────┘
                   │
                   ▼
        ┌──────────────────────┐
        │ Final Layer Norm     │  (B, L, D)
        └──────────┬───────────┘
                   │
                   ▼
    ┌─────────────────────────┐
    │ Output Projection       │  (B, L, V)
    │ (Linear to Vocab Size)  │
    └─────────────────────────┘
                   │
                   ▼
    ┌─────────────────────────┐
    │ Logits (B, L, V)        │
    └─────────────────────────┘
```

**Key Design Decisions:**
- **Pad Masking**: Sentences can be any range beetwen `0 <= len(text) <= context_length`, because of padding and padding masks used in computing attention
- **Gradient Checkpointing**: Memory-efficient training on limited GPUs (VRAM efficiency)
- **Self written attention layer**: for educational purposes
- **GPT-2 design**: project is implementation of GPT-2 model from open-AI

### Dataset

The project uses **Shakespeare text** (or any custom text corpus). The dataset pipeline:

1. **Raw Text** → Tokenization (BPE) → NumPy Arrays
2. **ShakespeareDataset**: Fixed-length sliding window
3. **ShakespeareDatasetWithStride**: Optimized with configurable stride and padding (if `stride=1` it is not the same as dataset `ShakespeareDataset`)

**Data Loading Example:**
```python
dataset = ShakespeareDataset(
    tokens_path="dataset/input_tokens.npy",
    context_len=256
)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
```

### Training

**Training Configuration Options:**
- **Optimizers**: AdamW with weight decay
- **Learning Rate Schedule**: Linear warmup → Cosine annealing
- **Gradient Accumulation**: Effective batch size > hardware batch size
- **Mixed Precision**: Automatic casting for faster computation
- **Early Stopping**: Monitor validation loss with patience
- **Checkpoint Management**: Save best models automatically

**Preset Configurations:**
- **gpt2-mini**: 384-dim, 6 layers, 256 context (87M parameters approx.)
- **gpt2**: 768-dim, 12 layers, 1024 context (124M parameters)

---

## Core Components

### 1. **Model: WrexGPT** (`src/Model.py`)

The main model that ties all components together.

```python
class WrexGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.gradient_checkpointing

        # Token embedding layer
        self.embedding = nn.Embedding(
            config.vocab_size,
            config.dim_embedded
        )
        
        # Learned positional embeddings
        self.positional_encoding = PositionalEncoding(
            config.context_length,
            config.dim_embedded
        )

        # Stack of transformer decoder blocks
        self.transformers = nn.ModuleList([
            TransformerDecoder(
                dim_embedded=config.dim_embedded,
                context_length=config.context_length,
                num_heads=config.num_heads,
                dropout=config.dropout
            )
            for _ in range(config.layers)
        ])

        # Output projection to vocabulary
        self.out_ln = nn.LayerNorm(config.dim_embedded)
        self.out_projection = nn.Linear(
            config.dim_embedded,
            config.vocab_size
        )

    def forward(self, x):
        # x: (batch_size, context_length)
        # Returns: logits (batch_size, context_length, vocab_size)
        
        batch_size, sentence_length = x.shape
        
        # Create padding mask for attention
        keep = (x != self.config.padding_token)
        keep2d = keep.unsqueeze(2) & keep.unsqueeze(1)
        mask_pad = ~keep2d.unsqueeze(1)
        
        # Embed tokens and add positional encoding
        x = self.embedding(x)
        pos_enc = self.positional_encoding(sentence_length, x.device)
        x = x + pos_enc
        
        # Pass through transformer stack
        for transformer in self.transformers:
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    transformer, x, mask_pad,
                    use_reentrant=False
                )
            else:
                x = transformer(x, mask_pad)
        
        # Project to vocabulary
        x = self.out_ln(x)
        x = self.out_projection(x)
        
        return x
```

**Key Features:**
- **Padding Mask**: Prevents gradient flow from padding tokens
- **Gradient Checkpointing**: Reduces memory usage at training time
- **Custom Weight Initialization**: Scaled initialization for residual connections
- **Device Agnostic**: Works on CPU or GPU

---

### 2. **Tokenizer** (`src/Data/Tokenizer.py`)

Handles text encoding/decoding using GPT-2's BPE tokenization.

```python
class Tokenizer:
    def __init__(self, padding_token=50257):
        self.padding_token = padding_token
        
        # Load GPT-2 base tokenizer
        gpt2_base_tokenizer = tiktoken.encoding_for_model("gpt2")
        
        # Add custom padding token
        self.tokenizer = tiktoken.Encoding(
            name="gpt2-extra-tokens",
            pat_str=gpt2_base_tokenizer._pat_str,
            mergeable_ranks=gpt2_base_tokenizer._mergeable_ranks,
            special_tokens={
                **gpt2_base_tokenizer._special_tokens,
                "<|pad|>": self.padding_token
            }
        )

    def encode(self, text: str) -> list:
        """Convert text to token IDs"""
        cleared_text = ftfy.fix_text(text)
        cleared_text = cleared_text.replace("\r\n", "\n")
        tokens = self.tokenizer.encode(cleared_text)
        return tokens

    def decode(self, tokens: list) -> str:
        """Convert token IDs back to text"""
        return self.tokenizer.decode(tokens)
```

**Features:**
- **BPE Encoding**: Byte-pair encoding compatible with OpenAI's GPT-2
- **Text Cleaning**: Uses ftfy to normalize unicode and line endings
- **Custom Tokens**: Added padding token for efficient batch processing

---

### 3. **Dataset Classes** (`src/Data/Dataset.py`)

Two implementations for different use cases:

```python
class ShakespeareDataset(torch.utils.data.Dataset):
    """Standard sliding window dataset"""
    def __init__(self, tokens_path, context_len, tokens=None):
        if tokens is None:
            self.tokens = np.load(tokens_path, mmap_mode="r")
        else:
            self.tokens = tokens
        self.context_len = context_len

    def __len__(self):
        return len(self.tokens) - self.context_len - 1

    def __getitem__(self, idx):
        x = torch.tensor(
            self.tokens[idx:idx + self.context_len],
            dtype=torch.long
        )
        y = torch.tensor(
            self.tokens[idx+1:idx + self.context_len + 1],
            dtype=torch.long
        )
        return x, y


class ShakespeareDatasetWithStride(torch.utils.data.Dataset):
    """Strided dataset with automatic padding"""
    def __init__(self, tokens_path, context_len, tokens=None,
                 stride=1, padding_token=50257):
        if tokens is None:
            self.tokens = np.load(tokens_path, mmap_mode="r")
        else:
            self.tokens = tokens
        self.context_len = context_len
        self.stride = int(stride)
        self.padding_token = padding_token

    def __len__(self):
        return math.ceil(len(self.tokens) / self.stride)

    def __getitem__(self, idx):
        current_index = idx * self.stride
        x = torch.tensor(
            self.tokens[current_index:current_index + self.context_len],
            dtype=torch.long
        )
        y = torch.tensor(
            self.tokens[current_index+1:current_index + self.context_len + 1],
            dtype=torch.long
        )
        
        # Pad to context_len if necessary
        if x.shape[0] != self.context_len:
            padding_needed = self.context_len - x.shape[0]
            x = torch.nn.functional.pad(
                x, (0, padding_needed),
                value=self.padding_token
            )
        
        if y.shape[0] != self.context_len:
            padding_needed = self.context_len - y.shape[0]
            y = torch.nn.functional.pad(
                y, (0, padding_needed),
                value=self.padding_token
            )
        
        return x, y
```
`ShakespeareDataset` **Features :**:
- **Memory Mapping**: Efficient handling of large datasets
- **Automatic and efficient `stride=1`**: In dataset below there can be padding tokens, and `stride=1` is not optimized. Here `stride=1` always, and dataset perfectly split tokens, there is **no padding tokens** in vector here

`ShakespeareDatasetWithStride` **Features :**
- **Memory Mapping**: Efficient handling of large datasets
- **Automatic Padding**: Handles variable-length sequences, adding padding tokens if needed
- **Configurable Stride**: Use a fraction of data for faster iteration

---

### 4. **Multi-Head Attention** (`src/Layers/Attention.py`)

Core attention mechanism with causal masking:

```python
class MaskedMultiHeadAttention(nn.Module):
    def __init__(self, dim_in, dim_out, context_length,
                 num_heads=12, dropout=0.1, bias=False):
        super().__init__()
        
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.num_heads = num_heads
        self.context_length = context_length
        self.head_dim = dim_out // num_heads

        # Query, Key, Value projections
        self.w_query = nn.Linear(dim_in, dim_out, bias=bias)
        self.w_key = nn.Linear(dim_in, dim_out, bias=bias)
        self.w_value = nn.Linear(dim_in, dim_out, bias=bias)

        # Causal mask (lower triangular)
        self.register_buffer(
            "mask",
            torch.triu(
                torch.ones(context_length, context_length),
                diagonal=1
            )
        )

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.out_projection = nn.Linear(dim_out, dim_out)
        self.out_projection.RESIDUAL_INIT = 1

    def forward(self, x, padding_mask=None):
        batch_size, context_length, dim_in = x.shape
        
        # Project to Q, K, V
        q = self.w_query(x)  # (B, L, D)
        k = self.w_key(x)
        v = self.w_value(x)
        
        # Reshape for multi-head: (B, L, D) -> (B, L, H, D/H)
        q = q.view(batch_size, context_length, self.num_heads, self.head_dim)
        k = k.view(batch_size, context_length, self.num_heads, self.head_dim)
        v = v.view(batch_size, context_length, self.num_heads, self.head_dim)
        
        # Transpose to: (B, H, L, D/H)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        att_score = q @ k.transpose(2, 3)  # (B, H, L, L)
        att_score = att_score / math.sqrt(self.head_dim)
        
        # Apply causal mask
        mask = self.mask.bool()[0:context_length, 0:context_length]
        att_score = att_score.masked_fill_(mask, torch.finfo(att_score.dtype).min)
        
        # Apply padding mask if provided
        if padding_mask is not None:
            att_score = att_score.masked_fill_(padding_mask, torch.finfo(att_score.dtype).min)
        
        # Softmax and dropout
        att_score = self.softmax(att_score)
        att_score = self.dropout(att_score)
        
        # Apply attention to values
        context_vec = att_score @ v  # (B, H, L, D/H)
        
        # Reshape back: (B, H, L, D/H) -> (B, L, H, D/H)
        context_vec = context_vec.transpose(1, 2)
        
        # Concatenate heads: (B, L, D)
        context_vec = context_vec.contiguous().view(batch_size, context_length, self.dim_out)
        
        # Final projection
        context_vec = self.out_projection(context_vec)
        
        return context_vec
```

**Key Mechanisms:**
- **Scaled Dot-Product Attention**: Prevents gradient saturation
- **Causal Masking**: Prevents attending to future positions
- **Padding Mask**: Ignores padding tokens in attention
- **Multi-Head**: Multiple representation subspaces

---

### 5. **Transformer Decoder Block** (`src/Layers/Transformer.py`)

Single decoder layer combining attention and MLP:

```python
class TransformerDecoder(nn.Module):
    def __init__(self, dim_embedded, context_length, num_heads,
                 dropout=0.1, qkv_bias=False):
        super().__init__()
        
        self.dim_embedded = dim_embedded
        
        # Attention sublayer
        self.attention = MaskedMultiHeadAttention(
            dim_in=dim_embedded,
            dim_out=dim_embedded,
            context_length=context_length,
            num_heads=num_heads,
            dropout=dropout,
            bias=qkv_bias
        )
        
        # Feed-forward sublayer
        self.mlp = MultiLayerPerceptron(
            dim_in=dim_embedded,
            dim_hidden=dim_embedded * 4
        )
        
        # Layer normalizations
        self.ln1 = nn.LayerNorm(dim_embedded)
        self.ln2 = nn.LayerNorm(dim_embedded)
        
        # Dropout layers
        self.dropout_attn = nn.Dropout(dropout)
        self.dropout_mlp = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-norm residual connection for attention
        identity = x
        x = self.ln1(x)
        x = self.attention(x, mask)
        x = self.dropout_attn(x)
        x = x + identity
        
        # Pre-norm residual connection for MLP
        identity = x
        x = self.ln2(x)
        x = self.mlp(x)
        x = self.dropout_mlp(x)
        x = x + identity
        
        return x
```

**Features:**
- **Pre-Normalization**: Layer norm before layers (more stable)
- **Residual Connections**: Skip connections for gradient flow
- **Sublayer Design**: Independent attention and MLP

---

### 6. **Feed-Forward Network** (`src/Layers/MLP.py`)

Position-wise fully connected network:

```python
class MultiLayerPerceptron(nn.Module):
    def __init__(self, dim_in, dim_hidden):
        super().__init__()
        
        self.l1 = nn.Linear(dim_in, dim_hidden)
        self.gelu = nn.GELU()
        self.l2 = nn.Linear(dim_hidden, dim_in)
        self.l2.RESIDUAL_INIT = 1

    def forward(self, x):
        x = self.l1(x)      # (B, L, D) -> (B, L, 4D)
        x = self.gelu(x)    # GELU activation
        x = self.l2(x)      # (B, L, 4D) -> (B, L, D)
        return x
```

**Design:**
- **Expansion Factor**: 4× expansion mimics GPT-2 design
- **GELU Activation**: Smooth activation function used in transformers
- **Residual Initialization**: Scaled weights for residual connections

---

### 7. **Positional Encoding** (`src/Layers/PositionalEncoding.py`)

Learned positional embeddings:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, context_length, dim_embedded):
        super().__init__()
        self.position_embedding = nn.Embedding(
            context_length,
            dim_embedded
        )

    def forward(self, sentence_length, device):
        # Generate position indices: (0, 1, 2, ..., L-1)
        pos = torch.arange(sentence_length, device=device, dtype=torch.long)
        
        # Embed positions: (L,) -> (L, D)
        pos_emb = self.position_embedding(pos)
        
        return pos_emb
```

**Features:**
- **Learned Embeddings**: Unlike original transformer's sinusoidal encoding
- **Dynamic Length**: Handles variable sequence lengths
- **Device-Aware**: Automatically placed on correct device

---

### 8. **Configuration Classes** (`src/Utils/Config.py`)

Centralized configuration management:

```python
@dataclass
class ModelConfig:
    """GPT-2 Model Configuration"""
    dim_embedded: int          # Embedding dimension (usually 768)
    context_length: int        # Maximum sequence length (usually 1024)
    num_heads: int            # Number of attention heads (usually 12)
    layers: int               # Number of transformer layers (usually 12)
    dropout: float            # Dropout probability
    gradient_checkpointing: bool = True  # Memory optimization
    vocab_size: int = 50258   # GPT-2 vocabulary size
    padding_token: int = 50257  # Padding token ID

    @classmethod
    def from_preset(cls, preset: str):
        """Load preset configuration"""
        configs = {
            "gpt2-mini": {
                "dim_embedded": 384,
                "context_length": 256,
                "num_heads": 6,
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
        return cls(**configs[preset])


@dataclass
class TrainConfig:
    """Training Configuration"""
    total_steps: int              # Total training steps
    max_lr: float                # Peak learning rate
    batch_size: int              # Micro-batch size
    accumulation_batch_size: int  # Gradient accumulation batch size
    weight_decay: float           # L2 regularization
    grad_clip: float              # Gradient clipping threshold
    device: str                   # "cuda" or "cpu"
    warmup_steps: int            # LR warmup steps
    early_stopper_patience: int  # Early stopping patience
    use_amp: bool = None         # Automatic mixed precision

    @classmethod
    def from_preset(cls, preset: str):
        """Load preset training configuration"""
        configs = {
            "gpt2-mini": {
                "total_steps": 100000,
                "max_lr": 5e-4,
                "batch_size": 16,
                "accumulation_batch_size": 64,
                "weight_decay": 0.1,
                "grad_clip": 1.0,
                "warmup_steps": 2000,
                "early_stopper_patience": 15,
            },
            "gpt2": {
                "total_steps": 100000,
                "max_lr": 2.5e-4,
                "batch_size": 4,
                "accumulation_batch_size": 64,
                "weight_decay": 0.01,
                "grad_clip": 1.0,
                "warmup_steps": 2000,
                "early_stopper_patience": 20,
            }
        }
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(**configs[preset], device=device)
```

**Features:**
- **Pre-set configs**: you can choose from existing configs
- **Create your own**: you can create your own configs
- **Or modify existing!**

---

### 9. **Trainer** (`src/Utils/Trainer.py`)

Complete training loop with checkpointing:

```python
class GPT2Trainer:
    def __init__(self, model, config: TrainConfig, train_loader,
                 val_loader=None, earlystopper=None):
        self.model = model.to(config.device)
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.earlystopper = earlystopper
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.max_lr,
            weight_decay=config.weight_decay
        )
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=config.padding_token
        )
        
        # Learning rate scheduler
        warmup = LinearLR(self.optimizer, 
                         start_factor=1e-8, 
                         end_factor=1.0,
                         total_iters=config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer,
                                  T_max=config.total_steps - config.warmup_steps)
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[config.warmup_steps]
        )
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler(enabled=config.use_amp)
        
        # Training history
        self.history = {
            "test_loss": [],
            "val_loss": [],
            "test_acc": [],
            "val_acc": [],
        }

    def train_epochs(self, revive_mode=False):
        """Main training loop by epochs"""
        # Implementation handles loading checkpoints,
        # gradient accumulation, mixed precision, etc.
        pass

    def train_steps(self, revive_mode=False):
        """Main training loop by steps"""
        # Implementation handles loading checkpoints,
        # gradient accumulation, mixed precision, etc.
        pass
```

---

### 10. **Early Stopping** (`src/Utils/EarlyStopper.py`)

Prevents overfitting and saves best model by val_loss (but also can give any metric here that has attribute **less=better**):

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        
        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

    def step(self, val_loss, model, epoch):
        """Check if should stop training"""
        if val_loss < self.best_loss - self.min_delta:
            # Improvement found
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            torch.save(model.state_dict(), self.path)
            return False  # Continue training
        else:
            # No improvement
            self.counter += 1
            return self.counter >= self.patience  # Stop if patience exceeded
```

---

### 11. **Inference / Text Generation** (`src/Utils/Inference.py`)

Autoregressive text generation:

```python
class AutoregressiveGenerator:
    def __init__(self, model, config: ModelConfig, device):
        self.model = model.to(device)
        self.config = config
        self.tokenizer = Tokenizer()
        self.device = device

    @torch.no_grad()
    def generate(self, text: str, max_new_tokens: int = 50,
                 temperature=1.0, top_k=None, greedy=True):
        """Generate text autoregressively"""
        self.model.eval()
        
        # Encode input
        tokens = torch.tensor(
            self.tokenizer.encode(text),
            dtype=torch.long,
            device=self.device
        ).view(1, -1)  # Add batch dimension
        
        # Generate tokens one at a time
        for _ in range(max_new_tokens):
            # Handle context length
            if tokens.shape[1] <= self.config.context_length:
                cond_tokens = tokens
            else:
                cond_tokens = tokens[:, -self.config.context_length:]
            
            # Get next token logits
            logits = self.model(cond_tokens)  # (B, L, V)
            logits = logits[:, -1, :]         # (B, V)
            logits = logits / temperature
            
            # Apply top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            if greedy:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to sequence
            tokens = torch.cat((tokens, next_token), dim=1)
        
        # Decode and return
        return self.tokenizer.decode(tokens[0].tolist())
```

**Decoding Strategies:**
- **Greedy**: Always pick highest probability token
- **Sampling**: Sample from probability distribution
- **Top-k Filtering**: Only sample from k most likely tokens

---

## Usage

### 1. Tokenizing Data

```python
from src.Data import Tokenizer
import numpy as np

tokenizer = Tokenizer()

# Read text file
with open("dataset/input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# Tokenize
tokens = tokenizer.encode(text)

# Save as numpy array
arr = np.array(tokens, dtype=np.int32)
np.save("dataset/input_tokens.npy", arr)
```

### 2. Training

```python
from src.Model import WrexGPT
from src.Utils import ModelConfig, TrainConfig, GPT2Trainer
from src.Data import ShakespeareDataset
import torch
from torch.utils.data import DataLoader

# Load config
model_config = ModelConfig.from_preset("gpt2-mini")
train_config = TrainConfig.from_preset("gpt2-mini")

# Create model
model = WrexGPT(model_config)

# Load dataset
dataset = ShakespeareDataset("dataset/input_tokens.npy", 
                             context_len=256)
loader = DataLoader(dataset, 
                   batch_size=train_config.batch_size,
                   shuffle=True)

# Train
trainer = GPT2Trainer(model, train_config, loader)
trainer.train_epochs()
```

### 3. Text Generation

```python
from src.Model import WrexGPT
from src.Utils import ModelConfig, AutoregressiveGenerator

config = ModelConfig.from_preset("gpt2-mini")
model = WrexGPT(config)

generator = AutoregressiveGenerator(
    model, config,
    "cuda" if torch.cuda.is_available() else "cpu"
)

text = generator.generate(
    "Once upon a time",
    max_new_tokens=50,
    greedy=False,
    top_k=40
)
print(text)
```

### 4. Visualization

```python
import torch
import matplotlib.pyplot as plt

# Load training history
history = torch.load("checkpoints.pt")["history"]

# Plot loss
plt.figure(figsize=(12, 4))
plt.plot(history["train_loss"], label="Train")
plt.plot(history["val_loss"], label="Val")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.show()
```

---

## References

Based on the following seminal papers:

1. **"Attention Is All You Need"** (Vaswani et al., 2017)
   - Introduced the Transformer architecture
   - [Paper](https://arxiv.org/pdf/1706.03762)

2. **"Improving Language Understanding by Generative Pre-Training"** (Radford et al., 2018)
   - Early version: GPT-1
   - Demonstrated transfer learning for NLP
   - [Paper](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)

3. **"Language Models are Unsupervised Multitask Learners"** (Radford et al., 2019)
   - GPT-2: Scaling up transformer language models
   - Zero-shot learning capabilities
   - [Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

---

## Key Implementation Highlights

✅ **From-scratch Implementation**: No high-level abstractions; all components explicitly implemented
✅ **Production-Ready**: Checkpointing, mixed precision, gradient accumulation
✅ **Educational**: Extensively commented code with detailed explanations
✅ **Flexible**: Easy to modify architecture, hyperparameters, and training procedures
✅ **Efficient**: Gradient checkpointing, memory mapping for large datasets
✅ **Well-Structured**: Clean modular design with clear separation of concerns

---

## Project Structure

```
WrexGPT/
├── src/
│   ├── Model.py                 # Main WrexGPT model
│   ├── Data/
│   │   ├── Tokenizer.py        # BPE tokenization
│   │   └── Dataset.py          # Dataset implementations
│   ├── Layers/
│   │   ├── Attention.py        # Multi-head attention
│   │   ├── MLP.py              # Feed-forward network
│   │   ├── Transformer.py      # Transformer decoder block
│   │   └── PositionalEncoding.py  # Positional embeddings
│   └── Utils/
│       ├── Config.py           # Configuration classes
│       ├── Trainer.py          # Training loop
│       ├── EarlyStopper.py     # Early stopping
│       └── Inference.py        # Text generation
├── data_load.py                 # Data preprocessing script
├── train.py                     # Training entry point
├── generating.py                # Generation entry point
├── plotting.py                  # Visualization script
└── dataset/
    ├── input.txt               # Raw text data
    └── input_tokens.npy        # Tokenized data
```

---

## Author Notes

This implementation prioritizes **clarity and understanding** over performance optimization. Each component is designed to be:
- **Self-contained**: Can be understood in isolation
- **Modular**: Easy to swap implementations
- **Documented**: Extensive comments and examples
- **Debuggable**: Clear tensor shape tracking

