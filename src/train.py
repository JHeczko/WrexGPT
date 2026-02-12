import torch
from torch.utils.data import DataLoader
import numpy as np

from DataProcessing import ShakespeareDataset
from Model import WrexGPT
from Utils import ModelConfig, TrainConfig, GPT2Trainer, EarlyStopping

if __name__ == "__main__":
    gpt_config = ModelConfig.from_preset("gpt2-mini")
    train_config = TrainConfig.from_preset("gpt2-mini")

    model = WrexGPT(config = gpt_config)

    tokens = np.load("./dataset/input_tokens.npy")
    split = 0.9

    train_data = tokens[:int(split * len(tokens))]
    val_data = tokens[int(split * len(tokens)):]

    train_ds = ShakespeareDataset("", gpt_config.context_length, train_data)
    val_ds = ShakespeareDataset("", gpt_config.context_length, val_data)

    if torch.cuda.is_available():
        train_loader = DataLoader(dataset=train_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(dataset=train_ds, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_ds, batch_size=train_config.batch_size, shuffle=True)

    # updating number of iters in order to have 300 epochs
    epochs = 300
    print(len(train_loader)*epochs - train_config.warmup_steps)
    train_config.training_steps = len(train_loader)*epochs - train_config.warmup_steps

    print(gpt_config)
    print(train_config)

    earlystopper = EarlyStopping(patience=train_config.early_stopper_patience, path="./checkpoints/best_model.pt")

    trainer = GPT2Trainer(model=model, config=train_config, train_loader=train_loader, val_loader=val_loader, checkpoint_path="./checkpoints/checkpoint.pt")

    trainer.train(revive_mode=False)