from transformers import PretrainedConfig

#定义模型超参数
class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim:int = 256, #模型维度
            n_layers:int = 4, #Transformer层数
            n_heads:int = 8, # 注意力机制的头数
            n_kv_heads:int = 8, #键值头的数量
            vocab_size:int = 6144,# 词汇表大小
            hidden_dim:int = None, #前馈网络的隐藏层维度
            multiple_of:int = 64,   #前馈网络的维度必须是这个数的倍数，用于硬件提速
            norm_eps:float = 1e-5, #归一化层的eps，防止除零错误
            max_seq_len:int = 512, #最大序列长度
            dropout:float = 0.0, #dropout率
            flash_attn:bool = True, #是否使用flash attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.mutiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.droupout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)
