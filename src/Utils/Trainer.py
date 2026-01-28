import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .Config import TrainConfig

# @dataclass
# class TrainConfig:
#     epochs: int = 5
#     lr: float = 3e-4
#     weight_decay: float = 0.1
#     grad_clip: float = 1.0
#     device: str = "cuda" if torch.cuda.is_available() else "cpu"
#
#     # mixed precision
#     use_amp: bool = True
#
#     # logging
#     log_every: int = 50
#
#     # optional: scheduler
#     warmup_steps: int = 0


class GPTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        cfg: TrainConfig = TrainConfig()
    ):
        self.model = model.to(cfg.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.cfg = cfg

        # optimizer (AdamW like GPT-2)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay
        )

        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and "cuda" in cfg.device))

        # scheduler (opcjonalny warmup)
        self.global_step = 0
        self.scheduler = None
        if cfg.warmup_steps > 0:
            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=self._warmup_lr_lambda
            )

    def _warmup_lr_lambda(self, step: int):
        # linear warmup: 0 -> 1 over warmup_steps
        if step < self.cfg.warmup_steps:
            return float(step + 1) / float(self.cfg.warmup_steps)
        return 1.0

    # -------- core loss --------
    @staticmethod
    def compute_loss(logits: torch.Tensor, targets: torch.Tensor, padding_token=50257) -> torch.Tensor:
        """
        logits: (B, T, V)
        targets: (B, T)
        """
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.view(B * T, V),
            targets.view(B * T),
            ignore_index=padding_token,
        )
        return loss

    @staticmethod
    def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """
        token accuracy (nie super ważne w LM, ale bywa przydatne)
        """
        preds = logits.argmax(dim=-1)  # (B, T)
        correct = (preds == targets).float().mean().item()
        return correct

    # -------- epoch loops --------
    def train_epoch(self, epoch_idx: int) -> Dict[str, Any]:
        self.model.train()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        start_time = time.time()

        for batch_idx, (x, y) in enumerate(self.train_loader):
            x = x.to(self.cfg.device, non_blocking=True)  # (B, T)
            y = y.to(self.cfg.device, non_blocking=True)  # (B, T)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                logits = self.model(x)  # (B, T, V)
                loss = self.compute_loss(logits, y)

            # backward
            self.scaler.scale(loss).backward()

            # grad clip
            if self.cfg.grad_clip is not None and self.cfg.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)

            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            # metrics
            acc = self.compute_accuracy(logits.detach(), y)
            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

            # logging
            if (batch_idx + 1) % self.cfg.log_every == 0:
                avg_loss = total_loss / n_batches
                ppl = math.exp(min(avg_loss, 20))
                elapsed = time.time() - start_time
                lr = self.optimizer.param_groups[0]["lr"]

                print(
                    f"[train] epoch={epoch_idx+1} step={self.global_step} "
                    f"batch={batch_idx+1}/{len(self.train_loader)} "
                    f"loss={avg_loss:.4f} ppl={ppl:.2f} acc={total_acc/n_batches:.3f} "
                    f"lr={lr:.2e} time={elapsed:.1f}s"
                )

        avg_loss = total_loss / max(1, n_batches)
        avg_acc = total_acc / max(1, n_batches)
        ppl = math.exp(min(avg_loss, 20))

        return {"loss": avg_loss, "ppl": ppl, "acc": avg_acc}

    @torch.no_grad()
    def validate_epoch(self, epoch_idx: int) -> Optional[Dict[str, Any]]:
        if self.val_loader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for x, y in self.val_loader:
            x = x.to(self.cfg.device, non_blocking=True)
            y = y.to(self.cfg.device, non_blocking=True)

            logits = self.model(x)
            loss = self.compute_loss(logits, y)
            acc = self.compute_accuracy(logits, y)

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_acc = total_acc / max(1, n_batches)
        ppl = math.exp(min(avg_loss, 20))

        print(f"[val]   epoch={epoch_idx+1} loss={avg_loss:.4f} ppl={ppl:.2f} acc={avg_acc:.3f}")
        return {"loss": avg_loss, "ppl": ppl, "acc": avg_acc}

    @torch.no_grad()
    def test_all(self) -> Optional[Dict[str, Any]]:
        if self.test_loader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for x, y in self.test_loader:
            x = x.to(self.cfg.device, non_blocking=True)
            y = y.to(self.cfg.device, non_blocking=True)

            logits = self.model(x)
            loss = self.compute_loss(logits, y)
            acc = self.compute_accuracy(logits, y)

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_acc = total_acc / max(1, n_batches)
        ppl = math.exp(min(avg_loss, 20))

        print(f"[test]  loss={avg_loss:.4f} ppl={ppl:.2f} acc={avg_acc:.3f}")
        return {"loss": avg_loss, "ppl": ppl, "acc": avg_acc}

    # -------- full training --------
    def train_all(self):
        best_val_loss = float("inf")
        best_state = None

        for epoch in range(self.cfg.epochs):
            train_metrics = self.train_epoch(epoch)
            val_metrics = self.validate_epoch(epoch)

            if val_metrics is not None and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                best_state = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
                print(f"✅ New best val loss: {best_val_loss:.4f}")

        # restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            print("🔁 Restored best validation checkpoint.")

        # final test
        self.test_all()
