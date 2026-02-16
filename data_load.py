import numpy as np
import tiktoken
import ftfy

from src.Data import Tokenizer

# do it only once to tokenize data and save it to numpy file
if __name__ == '__main__':
    tokenizer_class = Tokenizer()


    input_path = "dataset/input.txt"
    out_path = "dataset/input_tokens.npy"

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    tokens = tokenizer_class.encode(text)

    arr = np.array(tokens, dtype=np.int32)
    np.save(out_path, arr)

    print("tokens:", arr.shape)
