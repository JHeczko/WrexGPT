import math
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .Config import TrainConfig, ConfigGPT2


class GPT2Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig, train_loader, val_loader=None, checkpoint_path=""):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.loss_fn = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.max_lr)

        self.scaler = torch.amp.GradScaler(device=config.device, enabled=(config.use_amp and "cuda" in config.device))

        self.global_step = 0
        self.current_epoch = 0

        self.path = checkpoint_path

        self.total_iters = self.train_loader.__len__() * self.config.epochs

        warmup = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=self.total_iters-self.config.warmup_steps, eta_min=0)
        self.scheduler = SequentialLR(optimizer=self.optimizer, schedulers=[warmup, cosine], milestones=[self.config.warmup_steps])

    def load_checkpoint(self):
        checkpoint = torch.load(self.path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.config.use_amp and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.global_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]

        print(f"Checkpoint loaded from {self.path}")

    def __save_checkpoint(self):

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "scaler_state_dict": self.scaler.state_dict() if self.config.use_amp else None,
            "config": self.config.__dict__
        }

        torch.save(checkpoint, self.path)
        print(f"Checkpoint saved to {self.path}")

    def __calculate_loss(self, logits, y):
        # logits = (B, T, V)
        # y      = (B, T)
        B, T, V = logits.shape
        logits = logits.view(B * T, V)
        y = y.view(B * T)
        loss = self.loss_fn(logits, y, ignore_index=self.config.padding_token)
        return loss

    @torch.no_grad()
    def __calculate_accuracy(self, logits, y):
        # logits = (B, T, V)
        # y      = (B, T)
        preds = logits.argmax(dim=-1)  # (B, T)
        mask = (y != self.config.padding_token)

        if mask.sum() == 0:
            return 0.0

        correct = ((preds == y) & mask).float().sum()
        total = mask.float().sum()
        return (correct / total).item()

    def __train_epoch(self):
        running_loss = 0.0
        running_acc = 0.0
        total_batches = 0

        self.model.train()

        for X, y in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)

            X = X.to(self.config.device, non_blocking=True)
            y = y.to(self.config.device, non_blocking=True)

            if self.config.use_amp:
                with torch.autocast(device_type="cuda"):
                    logits = self.model(X)
                    loss = self.__calculate_loss(logits, y)

                self.scaler.scale(loss).backward()

                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()

            else:
                logits = self.model(X)
                loss = self.__calculate_loss(logits, y)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            self.global_step += 1

            running_loss += loss.item()
            running_acc += self.__calculate_accuracy(logits.detach(), y)
            total_batches += 1

        self.current_epoch += 1

        avg_loss = running_loss / max(1,total_batches)
        avg_acc = running_acc / max(1,total_batches)
        ppl = math.exp(min(avg_loss, 20))

        return avg_loss, avg_acc, ppl

    def __validate_model(self):
        if self.val_loader is None:
            return None

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for x, y in self.val_loader:
            x = x.to(self.config.device, non_blocking=True)
            y = y.to(self.config.device, non_blocking=True)

            logits = self.model(x)
            loss = self.__calculate_loss(logits, y)
            acc = self.__calculate_accuracy(logits, y)

            total_loss += loss.item()
            total_acc += acc
            n_batches += 1

        avg_loss = total_loss / max(1, n_batches)
        avg_acc = total_acc / max(1, n_batches)
        ppl = math.exp(min(avg_loss, 20))

        return avg_loss, avg_acc, ppl

    def train(self): pass


class GPTTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        cfg: TrainConfig = TrainConfig()):
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