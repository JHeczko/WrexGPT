import torch
from torch.utils.data import DataLoader
import numpy as np

import math

from src.Data import ShakespeareDataset, ShakespeareDatasetWithStride
from src.Model import WrexGPT
from src.Utils import ModelConfig, TrainConfig, GPT2Trainer, EarlyStopping


if __name__ == "__main__":
    gpt_config = ModelConfig.from_preset("gpt2-mini")
    train_config = TrainConfig.from_preset("gpt2-mini")

    train_config.warmup_steps = 4000
    train_config.early_stopper_patience = 18
    train_config.max_lr = 2.5e-4
    train_config.weight_decay = 0.2

    model = WrexGPT(config = gpt_config)

    tokens = np.load("dataset/input_tokens.npy")
    split = 0.9

    train_data = tokens[:int(split * len(tokens))]
    val_data = tokens[int(split * len(tokens)):]

    # train_ds = ShakespeareDatasetWithStride("", gpt_config.context_length, train_data, stride=math.ceil(gpt_config.context_length/4), padding_token=train_config.padding_token)
    # val_ds = ShakespeareDatasetWithStride("", gpt_config.context_length, val_data, stride=math.ceil(gpt_config.context_length/8), padding_token=train_config.padding_token)

    train_ds = ShakespeareDataset("", context_len=gpt_config.context_length, tokens=train_data)
    val_ds = ShakespeareDataset("", context_len=gpt_config.context_length, tokens=val_data)

    if torch.cuda.is_available():
        train_loader = DataLoader(dataset=train_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True)
        val_loader = DataLoader(dataset=val_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True)
    else:
        train_loader = DataLoader(dataset=train_ds, batch_size=train_config.batch_size, shuffle=True)
        val_loader = DataLoader(dataset=val_ds, batch_size=train_config.batch_size, shuffle=True)


    print(gpt_config)
    print(train_config)

    print("Dataset token size: ", len(tokens))
    print("Train loader batches: ", len(train_loader))
    print("Val loader batches: ", len(val_loader))

    earlystopper = EarlyStopping(patience=train_config.early_stopper_patience, path="./best_model.pt")

    trainer = GPT2Trainer(model=model, config=train_config, train_loader=train_loader, val_loader=val_loader, checkpoint_path="./checkpoint.pt", earlystopper=earlystopper)

    #history = trainer.train_epochs(revive_mode=False)
    history = trainer.train_steps(revive_mode=False)
