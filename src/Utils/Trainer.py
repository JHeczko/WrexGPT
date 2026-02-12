import math

import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from Config import TrainConfig


class GPT2Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig, train_loader, val_loader=None, earlystoper=None, checkpoint_path=""):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.earlystopper = earlystoper

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.padding_token)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.max_lr)

        self.scaler = torch.amp.GradScaler(device=config.device, enabled=(config.use_amp and "cuda" in config.device))

        self.current_step = 0
        self.current_epoch = 0

        self.path = checkpoint_path

        self.history = {
            "test_loss": [],
            "test_acc": [],
            "test_ppl": [],
            "val_loss": [],
            "val_acc": [],
            "val_ppl": [],
        }

        steps_per_epoch = len(self.train_loader)
        self.epochs = math.ceil(self.config.training_steps / steps_per_epoch)
        self.total_steps = self.config.training_steps + self.config.warmup_steps

        warmup = LinearLR(self.optimizer, start_factor=1e-8, end_factor=1.0, total_iters=config.warmup_steps)
        cosine = CosineAnnealingLR(self.optimizer, T_max=self.total_steps-self.config.warmup_steps, eta_min=0)
        self.scheduler = SequentialLR(optimizer=self.optimizer, schedulers=[warmup, cosine], milestones=[self.config.warmup_steps])

    def __load_checkpoint(self):
        checkpoint = torch.load(self.path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scheduler and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.config.use_amp and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if checkpoint["earlystopper_state_dict"] is not None and self.earlystopper is not None:
            self.earlystopper.load_state_dict(checkpoint["earlystopper_state_dict"])

        self.history = checkpoint["history"]

        self.current_step = checkpoint["global_step"]
        self.current_epoch = checkpoint["current_epoch"]

        print(f"Checkpoint loaded from {self.path}")

    def __save_checkpoint(self):

        checkpoint = {
            "epoch": self.current_epoch,
            "global_step": self.current_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
            "earlystopper_state_dict": self.earlystopper.state_dict() if self.earlystopper else None,
            "scaler_state_dict": self.scaler.state_dict() if self.config.use_amp else None,
            "history": self.history,
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
        loss = self.loss_fn(logits, y)
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

            self.current_step += 1

            running_loss += loss.item()
            running_acc += self.__calculate_accuracy(logits.detach(), y)
            total_batches += 1

            if self.current_step%self.config.info_decay == 0:
                val_loss, val_acc, val_ppl = self.__validate_model()
                print(
                    f"[step {self.current_step:>7}] "
                    f"test_loss={running_loss / max(1,total_batches):.4f} | "
                    f"test_acc={running_acc / max(1,total_batches):.4f} | "
                    f"val_loss={val_loss:.4f} | "
                    f"val_acc={val_acc:.4f} | "
                    f"val_ppl={val_ppl:.2f}"
                )

            if self.current_step == self.total_steps:
                break



        avg_loss = running_loss / max(1,total_batches)
        avg_acc = running_acc / max(1,total_batches)
        ppl = math.exp(min(avg_loss, 20))

        return avg_loss, avg_acc, ppl

    @torch.no_grad()
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

    def train(self, revive_mode=False):

        if revive_mode:
            self.__load_checkpoint()

        print("=" * 80)
        print("Starting training...")
        print(f"Total epochs: {self.epochs}")
        print("=" * 80)

        for epoch in range(self.current_epoch, self.epochs):

            train_loss, train_acc, train_ppl = self.__train_epoch()
            val_loss, val_acc, val_ppl = self.__validate_model()

            self.history["test_loss"].append(train_loss)
            self.history["test_acc"].append(train_acc)
            self.history["test_ppl"].append(train_ppl)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["val_ppl"].append(val_ppl)

            # ======== PRINT RESULTS ========
            print(f"\nEpoch [{epoch + 1}/{self.epochs}]")
            print("-" * 60)
            print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | Train PPL: {train_ppl:.2f}")
            print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f} | Val   PPL: {val_ppl:.2f}")
            print(f"LR: {self.optimizer.param_groups[0]['lr']:.6e}")
            print("-" * 60)

            # ======== EARLY STOPPING ========
            if self.earlystopper is not None:
                should_stop = self.earlystopper.step(val_loss, self.model, epoch)

                if should_stop:
                    print("\nEarly stopping triggered.")
                    break

            # ======== UPDATE EPOCHS ========
            self.current_epoch += 1

            # ======== SAVE CHECKPOINT ========
            self.__save_checkpoint()


        print("\nTraining finished.")
        return self.history