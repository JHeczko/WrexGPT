import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, path="best_model.pt"):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path

        self.best_loss = float("inf")
        self.counter = 0
        self.best_epoch = 0

        self.best_model = None

    def step(self, val_loss, model, epoch):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch

            torch.save(model.state_dict(), self.path)
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience

    def state_dict(self):
      return {
          "patience": self.patience,
          "min_delta": self.min_delta,
          "path": self.path,
          "best_loss": self.best_loss,
          "counter": self.counter,
          "best_epoch": self.best_epoch,
      }

    def load_state_dict(self, state_dict):
        self.patience = state_dict["patience"]
        self.min_delta = state_dict["min_delta"]
        self.path = state_dict["path"]
        self.best_loss = state_dict["best_loss"]
        self.counter = state_dict["counter"]
        self.best_epoch = state_dict["best_epoch"]