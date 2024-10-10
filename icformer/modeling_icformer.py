import math
import warnings

import torch
import torch.nn as nn

from transformers.activations import ACT2FN
from transformers.modeling_utils import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions
from transformers.utils import logging

from icformer.configuration import ICFormerConfig

logger = logging.get_logger(__name__)

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # Assign the last few positional embeddings to the query.
    q_embed = (q * cos[:,:,-q.shape[-2]:,:]) + (rotate_half(q) * sin[:,:,-q.shape[-2]:,:]) 
    # q_embed = (q * cos[:,:,:q.shape[-2]:,:]) + (rotate_half(q) * sin[:,:,:q.shape[-2],:])
    k_embed = (k * cos[:,:,:k.shape[-2],:]) + (rotate_half(k) * sin[:,:,:k.shape[-2],:])
    return q_embed, k_embed

def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class ICFormerRotaryEmbedding(nn.Module):
    """Copyed from LlamaRotaryEmbedding"""
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None, scaling_factor=1.0):
        super().__init__()
        self.scaling_factor = scaling_factor
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        # device_type = device_type if isinstance(device_type, str) else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
    
class ICFormerLinearScalingRotaryEmbedding(ICFormerRotaryEmbedding):
    """Copyed from LlamaLinearScalingRotaryEmbedding """

    def forward(self, x, position_ids):
        # difference to the original RoPE: a scaling factor is aplied to the position ids
        position_ids = position_ids.float() / self.scaling_factor
        cos, sin = super().forward(x, position_ids)
        return cos, sin


class ICFormerDynamicNTKScalingRotaryEmbedding(ICFormerRotaryEmbedding):
    """Copyed from LlamaDynamicNTKScalingRotaryEmbedding"""

    def forward(self, x, position_ids):
        # difference to the original RoPE: inv_freq is recomputed when the sequence length > original length
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (
                base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: this may break with compilation

        cos, sin = super().forward(x, position_ids)
        return cos, sin

class ICFormerAttention(nn.Module):
    """Referenced LlamaAttention and Blip2QFormerMultiHeadAttention"""
    def __init__(self, config: ICFormerConfig, layer_idx: int, is_cross_attention=False):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.context_hidden_size = config.context_hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        
        if (self.num_key_value_groups * self.num_key_value_heads) != self.num_heads:
            raise ValueError(
                f"num_heads must be divisible by num_key_value_heads (got `num_heads`: {self.num_heads}"
                f" and `num_key_value_heads`: {self.num_key_value_heads})."
            )

        if is_cross_attention:
            self.k_proj = nn.Linear(self.context_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
            self.v_proj = nn.Linear(self.context_hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        else:
            self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
            self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

        self._init_rope()

    def _init_rope(self):
        if self.config.rope_scaling is None:
            self.rotary_emb = ICFormerRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    base=self.rope_theta,
                )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = ICFormerLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = ICFormerDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    
    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        context_hidden_states=None,
        context_attention_mask=None,
        output_attentions=False,
    ):  
        is_cross_attention = context_hidden_states is not None
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        if is_cross_attention:
            key_states = self.k_proj(context_hidden_states)
            value_states = self.v_proj(context_hidden_states)
            attention_mask = context_attention_mask

            context_bsz, context_len, _ = context_hidden_states.size()
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(context_bsz, context_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(context_bsz, context_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        else:
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)
            
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        
        cos, sin = self.rotary_emb(value_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        # if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
        #     raise ValueError(
        #         f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
        #         f" {attn_output.size()}"
        #     )

        attn_output = attn_output.transpose(1, 2).contiguous()
        if is_cross_attention:
            attn_output = attn_output.reshape(context_bsz, q_len, self.hidden_size)
        else:
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights
    
class ICFormerMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
class ICFormerRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        CRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

class ICFormerLayer(nn.Module):
    def __init__(self, config: ICFormerConfig, layer_idx: int, is_cross_attention: bool=True):
        super().__init__()
        self.hidden_size = config.hidden_size
        if is_cross_attention:
            self.cross_attn = ICFormerAttention(config=config, layer_idx=layer_idx, is_cross_attention=True)
            self.cross_input_layernorm = ICFormerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.context_layernorm = ICFormerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.has_cross_attention = True
        else:
            self.self_attn = ICFormerAttention(config=config, layer_idx=layer_idx, is_cross_attention=False)
            self.input_layernorm = ICFormerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.has_cross_attention = False

        self.post_attention_layernorm = ICFormerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = ICFormerMLP(config=config)
        # self.cross_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.context_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)
        # self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context_hidden_states=None,
        context_attention_mask=None,
        output_attentions=False,
    ):
        self_attn_weights, cross_attn_weights = None, None
        
        if self.has_cross_attention:
            # Cross Attention
            residual = hidden_states
            hidden_states, cross_attn_weights = self.cross_attention_forward(hidden_states, context_hidden_states, context_attention_mask, output_attentions)
            hidden_states = residual + hidden_states
        else:
            # Self Attention
            residual = hidden_states
            hidden_states, self_attn_weights = self.self_attention_forward(hidden_states, attention_mask, output_attentions)
            hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        return outputs
    
    def self_attention_forward(
        self,
        hidden_states,
        attention_mask=None,
        output_attentions=False,
    ):
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=self.prepare_position_ids(hidden_states),
            output_attentions=output_attentions,
        )
        return hidden_states, self_attn_weights
    
    def cross_attention_forward(
        self,
        hidden_states,
        context_hidden_states=None,
        context_attention_mask=None,
        output_attentions=False,
    ):
        cross_attn_weights = None
        if self.has_cross_attention and context_hidden_states is not None:
            hidden_states = self.cross_input_layernorm(hidden_states)
            context_hidden_states = self.context_layernorm(context_hidden_states) # beta-settings
            hidden_states, cross_attn_weights = self.cross_attn(
                hidden_states=hidden_states,
                context_hidden_states=context_hidden_states,
                context_attention_mask=context_attention_mask,
                position_ids=self.prepare_position_ids(hidden_states, context_hidden_states),
                output_attentions=output_attentions,
            )
        return hidden_states, cross_attn_weights

    def prepare_position_ids(self, query_embeds, context_hidden_states=None):
        if context_hidden_states is not None:
            max_seq_len = context_hidden_states.shape[1]
        else:
            max_seq_len = query_embeds.shape[1]
        position_ids = torch.arange(0, max_seq_len, device=query_embeds.device).unsqueeze(0)
        return position_ids.to(device=self.mlp.up_proj.weight.device)

class ICFormerPreTrainedModel(PreTrainedModel):
    config_class = ICFormerConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ICFormerLayer"]
    _supports_flash_attn_2 = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ICFormerModel):
            module.encoder.gradient_checkpointing = value

class ICFormerEncoder(nn.Module):
    def __init__(self, config:ICFormerConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [ICFormerLayer(config, layer_idx, layer_idx % config.cross_attention_frequency == 0) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = ICFormerRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False
    
    def forward(
        self,
        query_embeds,
        attention_mask=None,
        context_hidden_states=None,
        context_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions else None

        hidden_states = query_embeds

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                def create_custom_forward(module):
                    def custom_forward(*args, **kwargs):
                        return module(*args, **kwargs)
                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer), 
                    hidden_states, 
                    attention_mask, 
                    context_hidden_states, 
                    context_attention_mask,
                    output_attentions,
                )
            else:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask,
                    context_hidden_states,
                    context_attention_mask,
                    output_attentions,
                )
            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if layer.has_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        hidden_states = hidden_states.to(device=self.norm.weight.device)
        hidden_states = self.norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )
    
class ICFormerModel(ICFormerPreTrainedModel):
    def __init__(self, config:ICFormerConfig):
        super().__init__(config)
        self.config = config
        self.encoder = ICFormerEncoder(config)
        self.post_init()
    
    def forward(
        self,
        query_embeds,
        attention_mask=None,
        context_hidden_states=None,
        context_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        causal_self_attention=True,
        return_dict=None,
    ):
        if attention_mask is None:
            attention_mask = torch.ones(query_embeds.shape[:2], device=self.device)

        if context_attention_mask is None:
            context_attention_mask = torch.ones(context_hidden_states.shape[:2], device=self.device)

        attention_mask = self.get_extended_attention_mask(attention_mask, causal_self_attention=causal_self_attention)
        context_attention_mask = self.get_extended_attention_mask(context_attention_mask)
        
        return self.encoder(
            query_embeds=query_embeds,
            attention_mask=attention_mask,
            context_hidden_states=context_hidden_states,
            context_attention_mask=context_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def get_extended_attention_mask(
        self,
        attention_mask:torch.Tensor,
        causal_self_attention=False,
    ) -> torch.Tensor:
        """
        Makes broadcastable attention so that future and masked tokens are ignored.
        """
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        if causal_self_attention:
            batch_size, seq_length = attention_mask.shape
            seq_ids = torch.arange(seq_length, device=self.device)
            causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
            causal_mask = causal_mask.to(attention_mask.dtype)
            # make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            extended_attention_mask = causal_mask[:, None, :, :] * extended_attention_mask
            
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(self.dtype).min
        return extended_attention_mask.to(self.device)
    
    def enable_input_require_grads(self):
        pass
