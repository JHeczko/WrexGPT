from Model import WrexGPT
from Utils.Config import *
from Utils.Inference import AutoregressiveGenerator

import torch

if __name__=="__main__":
    config = ModelConfig.from_preset("gpt2-mini")
    model = WrexGPT(config)

    generator = AutoregressiveGenerator(model,config, "cpu")

    batch = torch.tensor([[1,2,3, config.padding_token,config.padding_token,config.padding_token,config.padding_token],[1,2,3, 4,5,config.padding_token,config.padding_token]], dtype=torch.long)
    model(batch)

    text = "I'm, whom i am"

    genereted_text = generator.generate(text, max_new_tokens=10)

    print(genereted_text)

