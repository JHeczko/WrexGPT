import math

from tqdm.auto import tqdm
import torch
from torch import nn
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

from .Config import TrainConfig


class GPT2Trainer:
    def __init__(self, model: nn.Module, config: TrainConfig, train_loader, val_loader=None, earlystoper=None, checkpoint_path=""):
        self.model = model
        self.model.to(config.device)

        self.config = config

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.earlystopper = earlystoper

        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.config.padding_token)

        if self.config.device == "cuda":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.max_lr, fused=True)
        else:
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=config.max_lr, fused=False)

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

        self.epochs = self.config.epochs

        if (self.config.accumulation_batch_size % self.config.batch_size != 0) or (self.config.accumulation_batch_size < self.config.batch_size):
            raise ValueError("Accumulated batch size has to be divisible by batch size and accumulated batch size cannot be less than batch size.")
        self.accumulation_step = self.config.accumulation_batch_size // self.config.batch_size

        # calculating all steps based on dataloader len and accumulation step size
        self.total_steps = math.ceil((len(self.train_loader)/self.accumulation_step)*self.epochs)
        training_steps = self.total_steps - self.config.warmup_steps


        # settuping schedulers
        if (self.config.accumulation_batch_size % self.config.batch_size != 0) or (self.config.accumulation_batch_size < self.config.batch_size):
            raise ValueError("Accumulated batch size has to be divisible by batch size and accumulated batch size cannot be less than batch size.")


        warmup = LinearLR(self.optimizer,
                  start_factor=1e-8,
                  end_factor=1.0,
                  total_iters=self.config.warmup_steps)

        cosine = CosineAnnealingLR(self.optimizer,
                                   T_max= training_steps,
                                   eta_min=0)

        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, cosine],
            milestones=[self.config.warmup_steps]
        )

    def __load_checkpoint(self):
        checkpoint = torch.load(self.path, map_location=self.config.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        self.config.__dict__.update(checkpoint["config"])

        if self.scheduler and checkpoint["scheduler_state_dict"] is not None:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        if self.config.use_amp and checkpoint["scaler_state_dict"] is not None:
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        if checkpoint["earlystopper_state_dict"] is not None and self.earlystopper is not None:
            self.earlystopper.load_state_dict(checkpoint["earlystopper_state_dict"])

        self.history = checkpoint["history"]

        self.path = checkpoint["path"]
        self.accumulation_step = checkpoint["accumulation_step"]

        self.current_step = checkpoint["current_step"]
        self.current_epoch = checkpoint["current_epoch"]

        print(f"Checkpoint loaded from {self.path}")

    def __save_checkpoint(self):

        checkpoint = {
            "current_epoch": self.current_epoch,
            "current_step": self.current_step,
            "accumulation_step": self.accumulation_step,
            "path": self.path,
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
        self.optimizer.zero_grad(set_to_none=True)

        for i,(X, y) in enumerate(tqdm(self.train_loader, desc=f"Training {self.current_epoch+1}", leave=False)):
            X,y = X.to(self.config.device, non_blocking=True), y.to(self.config.device, non_blocking=True)

            if self.config.use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = self.model(X)
                    raw_loss = self.__calculate_loss(logits, y)
                    loss = raw_loss / self.accumulation_step

                self.scaler.scale(loss).backward()

                # doing gradient step only after "accumulation_step"
                if (i+1)%self.accumulation_step == 0 or (i+1) >= len(self.train_loader):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=self.config.grad_clip)

                    self.scaler.step(self.optimizer)
                    if self.scheduler is not None:
                        self.scheduler.step()

                    self.scaler.update()

                    # gradient reset
                    self.optimizer.zero_grad(set_to_none=True)

                    self.current_step += 1
            else:
                logits = self.model(X)
                raw_loss = self.__calculate_loss(logits, y)
                loss = raw_loss / self.accumulation_step

                loss.backward()

                # doing gradient step only after "accumulation_step"
                if (i+1)%self.accumulation_step == 0 or (i+1) >= len(self.train_loader):

                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

                    self.optimizer.step()
                    if self.scheduler is not None:
                        self.scheduler.step()

                    # gradient reset
                    self.optimizer.zero_grad(set_to_none=True)

                    self.current_step += 1


            # LOSS AND ACC CALCULATION
            running_loss += raw_loss.item()
            running_acc += self.__calculate_accuracy(logits.detach(), y)
            total_batches += 1

            # ADDITIONAL VALIDATION
            if (self.current_step+1)%self.config.info_decay == 0:
                val_loss, val_acc, val_ppl = self.__validate_model()

                tqdm.tqdm.write(
                    f"[Step {self.current_step}] "
                    f"Avg Train Loss: {running_loss / total_batches:.4f} | "
                    f"Val Loss: {val_loss:.4f} | Val PPL: {val_ppl:.2f}"
                )

                self.model.train()



        # STATISTICS
        avg_loss = running_loss / max(1,total_batches)
        avg_acc = running_acc / max(1,total_batches)
        ppl = math.exp(min(avg_loss, 20))

        return avg_loss, avg_acc, ppl



    @torch.no_grad()
    def __validate_model(self):
        if self.val_loader is None:
            return 0.0, 0.0, 0.0

        self.model.eval()

        total_loss = 0.0
        total_acc = 0.0
        n_batches = 0

        for x, y in tqdm(self.val_loader, desc="Validating", leave=False):
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

        for epoch in tqdm(range(self.current_epoch, self.epochs), desc=f"Epoch {self.current_epoch+1}/{self.epochs}"):

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