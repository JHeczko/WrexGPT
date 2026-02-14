import torch
from torch.utils.data import DataLoader
import numpy as np

from DataProcessing import ShakespeareDataset, ShakespeareDatasetWithStride
from Model import WrexGPT
from Utils import ModelConfig, TrainConfig, GPT2Trainer, EarlyStopping

from matplotlib import pyplot as plt

if __name__ == "__main__":
    gpt_config = ModelConfig.from_preset("gpt2-test")
    train_config = TrainConfig.from_preset("gpt2-test")
    # train_config.epochs = 5
    # train_config.total_steps = -1

    model = WrexGPT(config = gpt_config)

    tokens = np.load("./dataset/input_tokens.npy")
    split = 0.9

    train_data = tokens[:int(split * len(tokens))]
    val_data = tokens[int(split * len(tokens)):]

    train_ds = ShakespeareDatasetWithStride("", gpt_config.context_length, train_data, stride=gpt_config.context_length/2, padding_token=train_config.padding_token)
    val_ds = ShakespeareDataset("", gpt_config.context_length, val_data)

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

    earlystopper = EarlyStopping(patience=train_config.early_stopper_patience, path="./checkpoints/best_model.pt")

    trainer = GPT2Trainer(model=model, config=train_config, train_loader=train_loader, val_loader=val_loader, checkpoint_path="./checkpoints/checkpoint.pt", earlystopper=earlystopper)

    #history = trainer.train_epochs(revive_mode=False)
    history = trainer.train_steps(revive_mode=True)

    plt.plot(history["lr"])
    plt.show()

    print(history)