import torch
import numpy as np

class ShakespeareDataset(torch.utils.data.Dataset):
    def __init__(self, tokens_path, context_len, tokens = None):
        if tokens is None:
            self.tokens = np.load(tokens_path, mmap_mode="r")
        else:
            self.tokens = tokens
        self.context_len = context_len

    def __len__(self):
        # cuz we have to left one word for last prediction
        return len(self.tokens) - self.context_len - 1

    def __getitem__(self, idx):
        # return x, y
        x = torch.tensor(self.tokens[idx:idx + self.context_len], dtype=torch.long)
        y = torch.tensor(self.tokens[idx+1:idx + self.context_len + 1], dtype=torch.long)
        return x,y


if __name__ == "__main__":
    dataset = ShakespeareDataset("../dataset/input_tokens.npy", 1024)
