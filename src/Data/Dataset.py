import torch
import numpy as np
import math

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


class ShakespeareDatasetWithStride(torch.utils.data.Dataset):
    def __init__(self, tokens_path, context_len, tokens = None, stride=1, padding_token=50257):
        if tokens is None:
            self.tokens = np.load(tokens_path, mmap_mode="r")
        else:
            self.tokens = tokens
        self.context_len = context_len
        self.stride = int(stride)
        self.padding_token = padding_token

    def __len__(self):
        # cuz we have to left one word for last prediction
        return math.ceil(len(self.tokens) / self.stride)

    def __getitem__(self, idx):
        # return x, y
        current_index = (idx*self.stride)
        x = torch.tensor(self.tokens[current_index: current_index + self.context_len], dtype=torch.long)
        y = torch.tensor(self.tokens[current_index+1: current_index + self.context_len + 1], dtype=torch.long)

        if x.shape[0] != self.context_len:
            padding_needed = self.context_len - x.shape[0]
            x = torch.nn.functional.pad(x, (0, padding_needed), value=self.padding_token)

        if y.shape[0] != self.context_len:
            padding_needed = self.context_len - y.shape[0]
            y = torch.nn.functional.pad(y, (0, padding_needed), value=self.padding_token)

        return x,y

if __name__ == "__main__":
    dataset = ShakespeareDataset("../dataset/input_tokens.npy", context_len=8, tokens=[i for i in range(563)])
    dataset_1 = ShakespeareDatasetWithStride("../dataset/input_tokens.npy", context_len=64, tokens=[i for i in range(563)], stride=56)

    ds_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    ds1_loader = torch.utils.data.DataLoader(dataset_1, batch_size=2, shuffle=False)

    for x,y in ds1_loader:
        print(f"X: {x}\nY: {y}")
