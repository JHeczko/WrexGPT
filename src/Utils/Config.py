class ConfigGPT:
    def __init__(self,  dim_embedded, vocab_size, context_length, num_heads, dropout, qkv_bias, device):
        self.__dim_embedded = dim_embedded
        self.__vocab_size = vocab_size
        self.__context_length = context_length
        self.__num_heads = num_heads


    @property
    def dim_embedded(self):
        return self.__dim_embedded

    @dim_embedded.setter
    def dim_embedded(self, dim_embedded):
        self.__dim_embedded = dim_embedded

    @dim_embedded.getter
    def dim_embedded(self):
        return self.__dim_embedded

    @property
    def vocab_size(self):
        return self.__vocab_size

    @vocab_size.setter
    def vocab_size(self, vocab_size):
        self.__vocab_size = vocab_size

    @vocab_size.getter
    def vocab_size(self):
        return self.__vocab_size

    @property
    def context_length(self):
        return self.__context_length

    @context_length.setter
    def context_length(self, context_length):
        self.__context_length = context_length

    @context_length.getter
    def context_length(self):
        return self.__context_length

    @property
    def num_heads(self):
        return self.__num_heads

    @num_heads.setter
    def num_heads(self, num_heads):
        self.__num_heads = num_heads

    @num_heads.getter
    def num_heads(self):
        return self.__num_heads


    @property
    def dropout(self):
        return self.__dropout

    @dropout.setter
    def dropout(self, dropout):
        self.__dropout = dropout

    @dropout.getter
    def dropout(self):
        return self.__dropout

    @property
    def qkv_bias(self):
        return self.__qkv_bias

    @qkv_bias.setter
    def qkv_bias(self, qkv_bias):
        self.__qkv_bias = qkv_bias

    @qkv_bias.getter
    def qkv_bias(self):
        return self.__qkv_bias

    @property
    def device(self):
        return self.__device

    @device.setter
    def device(self, device):
        self.__device = device

    @device.getter
    def device(self):
        return self.__device