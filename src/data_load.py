import numpy as np
import tiktoken
import ftfy

# do it only once to tokenize data and save it to numpy file
if __name__ == '__main__':
    gpt2_base_tokenizer = tiktoken.encoding_for_model("gpt2")

    tokenizer = tiktoken.Encoding(
        name="gpt2-extra-tokens",
        pat_str=gpt2_base_tokenizer._pat_str,
        mergeable_ranks=gpt2_base_tokenizer._mergeable_ranks,
        special_tokens={
            **gpt2_base_tokenizer._special_tokens,
            "<|pad|>": 50257
        }
    )

    input_path = "./dataset/input.txt"
    out_path = "./dataset/input_tokens.npy"

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    cleared_text = ftfy.fix_text(text)
    cleared_text = cleared_text.replace("\r\n", "\n")
    tokens = tokenizer.encode(cleared_text)

    arr = np.array(tokens, dtype=np.int32)
    np.save(out_path, arr)

    print("tokens:", arr.shape)
    print(text)
