import tiktoken
import ftfy

class Tokenizer:
    def __init__(self, padding_token=50257):
        super().__init__()
        self.padding_token = padding_token

        gpt2_base_tokenizer = tiktoken.encoding_for_model("gpt2")

        tokenizer = tiktoken.Encoding(
            name="gpt2-extra-tokens",
            pat_str=gpt2_base_tokenizer._pat_str,
            mergeable_ranks=gpt2_base_tokenizer._mergeable_ranks,
            special_tokens={
                **gpt2_base_tokenizer._special_tokens,
                "<|pad|>": self.padding_token
            }
        )

        self.tokenizer = tokenizer


    def encode(self, text: str):
        cleared_text = ftfy.fix_text(text)
        cleared_text = cleared_text.replace("\r\n", "\n")
        tokens = self.tokenizer.encode(cleared_text)
        return tokens

    def decode(self, tokens: list):
        return self.tokenizer.decode(tokens)