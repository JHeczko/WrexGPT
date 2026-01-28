import tiktoken
import torch
import numpy as np

if __name__ == "__main__":
    sens = ["Ala ma kota fajnego bardzo!", "fhjad asjk dhkashd kjhasjkd has", "fhjad asjk dhkashd kjhasjkd has", "fhjad asjk dhkashd kjhasjkd has", "!"]
    sen_tok = []
    context_len = 1024
    padding_token = 50256

    # 50256 padding token
    tokenizer = tiktoken.encoding_for_model("gpt2")
    for sen in sens:
        buf = tokenizer.encode(sen)
        buf.extend([50256 for _ in range(context_len - len(buf))])
        sen_tok.append(buf)

    tensor = torch.tensor(sen_tok)

    keep = (tensor != padding_token)  # (B, C) bool, True = token
    keep2d = keep.unsqueeze(2) & keep.unsqueeze(1)  # (B, C, C) bool True=keep
    mask_pad = ~keep2d  # True = maskuj
    # mask_pad = (batch_size, 1, context_len, context_len)
    mask_pad = mask_pad.unsqueeze(1)  # (B,1,C,C)

    print(mask_pad)
    print(tensor.shape)
    print(mask_pad.shape)