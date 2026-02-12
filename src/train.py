import torch
from torch.utils.data import DataLoader
from DataProcessing import ShakespeareDataset
from Model import WrexGPT
from Utils import ModelConfig, TrainConfig, GPT2Trainer, EarlyStopping

if __name__ == "__main__":
    gpt_config = ModelConfig.from_preset("gpt2-mini")
    train_config = TrainConfig.from_preset("gpt2-mini")

    print(gpt_config)
    print(train_config)

    model = WrexGPT(config = gpt_config)
    ds = ShakespeareDataset("./dataset/input_tokens.npy", gpt_config.context_length)

    generator = torch.Generator().manual_seed(42)
    train_ds, val_ds = torch.utils.data.random_split(
        ds,
        [0.9, 0.1],
        generator=generator
    )

    train_loader = DataLoader(dataset=train_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(dataset=val_ds, batch_size=train_config.batch_size, shuffle=True, pin_memory=True)


    earlystopper = EarlyStopping(patience=train_config.early_stopper_patience, path="./checkpoints/best_model.pt")

    trainer = GPT2Trainer(model=model, config=train_config, train_loader=train_loader, val_loader=val_loader, checkpoint_path="./checkpoints/checkpoint.pt")

    trainer.train(revive_mode=False)