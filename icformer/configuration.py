import torch

from transformers.configuration_utils import PretrainedConfig

class ICFormerConfig(PretrainedConfig):
    model_type = "icformer"
    def __init__(
        self,
        hidden_size=4096,
        num_hidden_layers=12,
        num_attention_heads=32,
        num_key_value_heads=None,
        num_query_tokens=128,
        intermediate_size=11008,
        hidden_act="silu",
        max_position_embeddings=2048,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        rope_scaling=None,
        attention_bias=False,
        cross_attention_frequency=1,
        context_hidden_size=4096,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        causal_attentin=True,
        tie_word_embeddings=False,
        attention_dropout=0.0,
        **kwargs,
    ):
        super().__init__()
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_query_tokens = num_query_tokens
        self.attention_dropout = attention_dropout

        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads
        
        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps

        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling

        self.cross_attention_frequency = cross_attention_frequency
        self.context_hidden_size = context_hidden_size

        self.attention_bias = attention_bias
        self.causal_attention = causal_attentin

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
    