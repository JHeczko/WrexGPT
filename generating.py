from src.Model import WrexGPT
from src.Utils import ModelConfig
from src.Utils import AutoregressiveGenerator

import torch

if __name__=="__main__":
    config = ModelConfig.from_preset("gpt2-mini")
    model = WrexGPT(config)

    generator = AutoregressiveGenerator(model,config, "cuda" if torch.cuda.is_available() else "cpu")

    text = "I'm, whom i am"

    genereted_text = generator.generate(text, max_new_tokens=30, greedy=False)

    print(genereted_text)

