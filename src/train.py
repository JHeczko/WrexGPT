import torch
from torch.utils.data import DataLoader
from DataProcessing import ShakespeareDataset
from model import WrexGPT
from src import Utils
from Utils import GPT2Trainer, TrainConfig

if __name__ == "__main__":
    config = Utils.ConfigGPT2(
        dim_embedded=12,
        context_length=12,
        num_heads=2,
        layers=2,
    )
    model = WrexGPT(config = config)

    ds = ShakespeareDataset("./dataset/input_tokens.npy", config.context_length)

    train_loader = DataLoader(dataset=ds, batch_size=512, shuffle=True)

    print(len(train_loader))

    trainer = GPT2Trainer(model=model, config=TrainConfig, train_loader=train_loader)

    trainer.train_epoch()