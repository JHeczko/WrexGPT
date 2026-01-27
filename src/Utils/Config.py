class ConfigGPT:
    def __init__(self,  dim_embedded, vocab_size, context_length, num_heads, layers):
        self.__dim_embedded = dim_embedded
        self.__vocab_size = vocab_size
        self.__context_length = context_length
        self.__num_heads = num_heads
        self.__layers = layers


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
    def layers(self):
        return self.__layers
    @layers.setter
    def layers(self, layers):
        self.__layers = layers
    @layers.getter
    def layers(self):
        return self.__layers