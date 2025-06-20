import copy
import json
import logging
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from transformers import AutoProcessor, AutoTokenizer

from typing import List, Optional, Tuple, Union
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.models.qwen2.modeling_qwen2 import (Qwen2Attention, Qwen2FlashAttention2, Qwen2SdpaAttention,
                                                      Qwen2DecoderLayer, Qwen2RMSNorm, Qwen2RotaryEmbedding)
from transformers import Qwen2Model, Qwen2Config, Qwen2ForCausalLM
from transformers.utils import is_flash_attn_greater_or_equal_2_10
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.idefics2.modeling_idefics2 import Idefics2VisionTransformer, Idefics2VisionConfig
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
)
from transformers import PretrainedConfig
from transformers import AutoModel, SwinForImageClassification
from transformers.image_utils import (
    IMAGENET_STANDARD_MEAN,
    IMAGENET_STANDARD_STD,)
from transformers.models.whisper.modeling_whisper import (WhisperAttention, WhisperSdpaAttention,
                                                          WhisperForConditionalGeneration)
from transformers import WhisperProcessor
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.utils import logging
logger = logging.get_logger(__name__)

        
import sys
sys.path.append("./")
from resampler import Resampler
from resampler import get_2d_sincos_pos_embed
from forward_baddmm import (forward_whisper_baddmm, forward_whisper_sdpa)
from transformers import SiglipVisionModel
from torchvision import transforms
from torch.nn.attention import SDPBackend, sdpa_kernel

from models.vae import DiagonalGaussianDistribution
from models.diffloss import DiffLoss

## init fixed baddmm forward
#WhisperAttention.forward = forward_whisper_baddmm
#WhisperSdpaAttention.forward = forward_whisper_sdpa

class LuckyConfig(Qwen2Config):
    model_type = "luckyllmv"
    default_vision_config = {
        "hidden_size": [1152, 1152, 1152, 1152, 1152, 1152],
        #"hidden_size": 1152,
        "image_size": 980,
        "intermediate_size": 4304,
        "model_type": "idefics2",
        "num_attention_heads": 16,
        "num_hidden_layers": 27,
        "patch_size": 14,
    }
    def __init__(self,
                 num_cross_layers: int = 5,
                 img_ids:int = 151665,
                 audio_ids:int = 151666,
                 query_num: int = 512,
                 audio_hidden_size: int = 768,
                 mlp_dim: int = 4096,
                 batch_vision_input: bool = True,
                 vision_model_name: str = "google/siglip-so400m-patch14-384",
                 audio_model_name: str = "openai/whisper-small",
                 **kwargs):
        self.num_cross_layers = num_cross_layers
        self.img_ids = img_ids
        self.audio_ids = audio_ids
        self.mlp_dim = mlp_dim
        self.audio_hidden_size = audio_hidden_size
        self.query_num = query_num
        self.vision_model_name = vision_model_name
        self.audio_model_name = audio_model_name
        self.batch_vision_input = batch_vision_input
        self.vision_config = Idefics2VisionConfig(**self.default_vision_config)
        super().__init__(**kwargs)

# Copied from transformers.models.llama.modeling_qwen -> Lucky Rotary Embedding
class LuckyRotaryEmbedding(Qwen2RotaryEmbedding):
    def __init__(self, config: Qwen2Config, device=None):
        super().__init__(config=config, device=device)

    @torch.no_grad()
    def forward(self, x, position_ids):
        assert position_ids.shape[0] == 2 and len(position_ids.shape) == 3
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, None, :, None].float().expand(2, position_ids.shape[1], -1, 1)
        position_ids_expanded = position_ids[:, :, None, :].float() # shape (2, bs, 1, positions)
        # Force float32 (see https://kkgithub.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float().to(x.device) @ position_ids_expanded.float()).transpose(2, 3)
            emb = torch.cat((freqs, freqs), dim=-1)
            emb = torch.cat([m[i % 2] for i, m in enumerate(emb.split([32, 32, 32, 32], dim=-1))], dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)

# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb_vision(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed

class Qwen2AttentionFixed(Qwen2Attention):
    """
    baddbmm Implement.
    """
    def __init__(self, config, layer_idx = None):
        super().__init__(config, layer_idx)
        self.rotary_emb = LuckyRotaryEmbedding(config=self.config)


class LuckyAttention(Qwen2AttentionFixed):
    """
    Multi-headed attention from 'Attention Is All You Need' paper. Modified to use sliding window attention: Longformer
    and "Generating Long Sequences with Sparse Transformers".
    """
    def __init__(self, config: LuckyConfig, layer_idx: Optional[int] = None):
        super().__init__(config, layer_idx)
        self.k_proj_addi = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.v_proj_addi = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=True)
        self.o_proj_addi = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)
        #self.gate = nn.Parameter(torch.zeros(self.hidden_size))
        self.gate = nn.Linear(self.hidden_size, 1, bias=False)
        self.rotary_emb = LuckyRotaryEmbedding(config=self.config)
        self.mask_token_layer = nn.Parameter(torch.normal(0, 0.02, size=(1, 1536)))
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        addi_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_img: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        mask_vision = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        bsz, q_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # Preallocate attn_weights for `baddbmm`(baddbmm faster)
        attn_weights = torch.empty(
            bsz * self.num_heads, q_len, key_states.shape[-2], dtype=query_states.dtype, device=query_states.device
        )
        attn_weights = torch.baddbmm(attn_weights,
                                     query_states.reshape(-1, q_len, self.head_dim),
                                     key_states.transpose(2, 3).view(-1, self.head_dim, key_states.shape[-2]),
                                     beta=0, alpha=1 / math.sqrt(self.head_dim)
                                     ).view(bsz, self.num_heads, q_len, key_states.shape[-2])
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.einsum("bhts,bhsd->bthd", attn_weights, value_states)

        if attn_output.size() != (bsz, q_len, self.num_heads, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        # custom cross attention
        if addi_hidden_states is not None:
            bsz, q_len, _ = hidden_states.size()
            
            _, k_v_len, _ = addi_hidden_states.size()

            if mask_vision is not None:
                mask_vision_layer = mask_vision.repeat(3,1).transpose(-2, -1).reshape(mask_vision.shape[0], -1).repeat(1,3).reshape(bsz, -1)
                addi_hidden_states[mask_vision_layer] = self.mask_token_layer
            key_states = self.k_proj_addi(addi_hidden_states)
            value_states = self.v_proj_addi(addi_hidden_states)

            key_states = key_states.view(bsz, k_v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, k_v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            cos_vision, sin_vision = self.rotary_emb(key_states, position_ids_img)
            key_states = apply_rotary_pos_emb_vision(key_states, cos_vision, sin_vision)
            # repeat k/v heads if n_kv_heads < n_heads
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
            
            # Preallocate attn_weights for `baddbmm`(baddbmm faster)
            attn_weights = torch.empty(
                bsz * self.num_heads, q_len, k_v_len, dtype=query_states.dtype, device=query_states.device
            )
            attn_weights = torch.baddbmm(attn_weights,
                                         query_states.reshape(-1, q_len, self.head_dim),
                                         key_states.transpose(2, 3).reshape(-1, self.head_dim, k_v_len),
                                         beta=0, alpha=1 / math.sqrt(self.head_dim)
                                         ).view(bsz, self.num_heads, q_len, k_v_len)

            if attn_weights.size() != (bsz, self.num_heads, q_len, k_v_len):
                raise ValueError(
                    f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                    f" {attn_weights.size()}"
                )
            #if self.training:
                #attn_weights = attn_weights + cross_attention_mask.unsqueeze(1)[:,:,-q_len:,:] if cross_attention_mask is not None else attn_weights
            # upcast attention to fp32
            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
            attn_output_cross = torch.einsum("bhts,bhsd->bthd", attn_weights, value_states)

            if attn_output_cross.size() != (bsz, q_len, self.num_heads, self.head_dim):
                raise ValueError(
                    f"`attn_output` should be of size {(bsz, q_len, self.num_heads, self.head_dim)}, but is"
                    f" {attn_output.size()}"
                )

            # attn_output_cross = attn_output_cross.transpose(1, 2).contiguous()
            attn_output_cross = attn_output_cross.reshape(bsz, q_len, self.hidden_size)

            attn_output_cross = self.o_proj_addi(attn_output_cross)
            gate_value = torch.nn.functional.sigmoid(self.gate(hidden_states)) # B N 1
            attn_output = attn_output_cross * gate_value + attn_output.clone()
            
        return attn_output, attn_weights, past_key_value
    

# Copied from transformers.models.llama.modeling_llama.repeat_kv
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


# Copied from transformers.models.mixtral.modeling_mixtral.MixtralSdpaAttention with Mixtral->Qwen2
class LuckySdpaAttention(LuckyAttention):
    """
    Lucky Attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LuckySdpaAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from Qwen2Attention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        addi_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_img: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
        mask_vision = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "Qwen2Model is using Qwen2SdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                addi_hidden_states=addi_hidden_states,
                attention_mask=attention_mask,
                cross_attention_mask=cross_attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        bsz, q_len, _ = hidden_states.size()
        _, k_v_len, _ = addi_hidden_states.size() if addi_hidden_states is not None else hidden_states.size()
        
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        if position_embeddings is None:
            logger.warning_once(
                "The attention layers in this model are transitioning from computing the RoPE embeddings internally "
                "through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed "
                "`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.46 `position_ids` will be "
                "removed and `position_embeddings` will be mandatory."
            )
            cos, sin = self.rotary_emb(value_states, position_ids)
        else:
            cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:  # no matter the length, we just slice it
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://git.homegu.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
        # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
        # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
        is_causal = True if causal_mask is None and q_len > 1 else False
        
        with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=causal_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)
        
        if addi_hidden_states is None:#or len(position_ids_img[0])==0:
            return attn_output, None, past_key_value
        ###################################################
        else:
            # custom cross attention

            if mask_vision is not None:
                mask_vision_layer = mask_vision.repeat(3,1).transpose(-2, -1).reshape(mask_vision.shape[0], -1).repeat(1,3).reshape(bsz, -1)
                addi_hidden_states[mask_vision_layer] = self.mask_token_layer
                
            key_states = self.k_proj_addi(addi_hidden_states)
            value_states = self.v_proj_addi(addi_hidden_states)

            key_states = key_states.view(bsz, k_v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, k_v_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

            #if self.training: causal_mask = cross_attention_mask.unsqueeze(1)[:,:,-q_len:,:]

            cos_vision, sin_vision = self.rotary_emb(key_states, position_ids_img)
            key_states = apply_rotary_pos_emb_vision(key_states, cos_vision, sin_vision)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)

            # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
            # Reference: https://github.com/pytorch/pytorch/issues/112577.
            if query_states.device.type == "cuda" and causal_mask is not None:
                #query_states = query_states.contiguous()
                key_states = key_states.contiguous()
                value_states = value_states.contiguous()

            # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
            # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
            # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
            is_causal = True if causal_mask is None and q_len > 1 else False
            
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attn_output_cross = torch.nn.functional.scaled_dot_product_attention(
                    query_states,
                    key_states,
                    value_states,
                    #attn_mask=causal_mask,
                    dropout_p=self.attention_dropout if self.training else 0.0,
                    is_causal=is_causal,
                )
            #print(torch.isnan(attn_output_cross).any())
            attn_output_cross = attn_output_cross.transpose(1, 2).contiguous()
            attn_output_cross = attn_output_cross.view(bsz, q_len, self.hidden_size)
            gate_value = torch.nn.functional.sigmoid(self.gate(hidden_states)) # B N 1
            attn_output_cross = self.o_proj_addi(attn_output_cross)
            attn_output = attn_output_cross * gate_value + attn_output.clone() #* (1 - gate_value)
            return attn_output, None, past_key_value
    
    
LUCKY_ATTENTION_CLASSES = {
    "eager": LuckyAttention,
    "flash_attention_2": LuckySdpaAttention,
    "sdpa": LuckySdpaAttention,
}


class Qwen2DecoderLayerFixed(Qwen2DecoderLayer):
    '''
    baddbmm Implement.
    '''
    def __init__(self, config: Qwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)

        self.self_attn = Qwen2AttentionFixed(config, layer_idx)


class LuckyDecoderLayer(Qwen2DecoderLayer):
    def __init__(self, config: LuckyConfig, layer_idx: int, addi_type="vision"):
        super().__init__(config, layer_idx)
        self.layer_idx = layer_idx
        self.self_attn = LUCKY_ATTENTION_CLASSES[config._attn_implementation](config, layer_idx)
        self.embed_dim = config.hidden_size
        self.query_num = config.query_num
        self.addi_type = addi_type
        if self.addi_type=="vision":
            try:
                self.addi_dim = config.vision_config.hidden_size[(layer_idx-6)//5]
            except:
                self.addi_dim = config.vision_config["hidden_size"][(layer_idx-6)//5]
            self.addi_adapter = self._build_adapter(self.embed_dim, self.addi_dim)
        elif self.addi_type=="audio":
            self.addi_dim = config.audio_hidden_size
            self.addi_adapter = Resampler(
                num_queries=config.query_num,
                embed_dim=self.embed_dim,
                num_heads=self.embed_dim // 128,
                kv_dim=self.addi_dim,
                adaptive=True,
                max_size=(1, 1500)
            )
        
    def _build_adapter(self, hidden_size, vision_dim=1024):
        mlp = [nn.LayerNorm(vision_dim, elementwise_affine=False),
               nn.Identity(),
               #nn.GELU(),
               nn.Linear(vision_dim, hidden_size)]
        return nn.Sequential(*mlp)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        addi_hidden: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_img: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[torch.Tensor] = None,
        mask_vision = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        if addi_hidden[0] is None:
            addi_hidden_states = None
        elif addi_hidden[1] is not None: # tgt size (audio)
            addi_hidden_states = self.addi_adapter(addi_hidden[0], addi_hidden[1])
            addi_hidden_states = self.input_layernorm(addi_hidden_states)
        else:
            addi_hidden_states = self.addi_adapter(addi_hidden[0])
            addi_hidden_states = self.input_layernorm(addi_hidden_states)

           # print(torch.isnan(addi_hidden_states).any())
        
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            addi_hidden_states=addi_hidden_states,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask if type(cross_attention_mask) != list else cross_attention_mask[(self.layer_idx-6)//5//2],
            position_ids=position_ids,
            position_ids_img=position_ids_img,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            mask_vision=mask_vision,
        )
        hidden_states = residual + hidden_states
        
        # Fully Connected
        residual = hidden_states
        #print(hidden_states)
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs
    
    
class LuckyModel(Qwen2Model):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`Qwen2DecoderLayer`]

    Args:
        config: LuckyConfig
    """
    config_class = LuckyConfig
    def __init__(self, config: LuckyConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.img_ids = config.img_ids
        self.audio_ids = config.audio_ids
        self.vocab_size = config.vocab_size
        self.num_cross_layers = config.num_cross_layers
        self.num_hidden_layers = config.num_hidden_layers

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.mask_token = nn.Parameter(torch.normal(0, 0.02, size=(1, 1536)))
        self.special_token = nn.Parameter(torch.rand(2, 1536))
        self.pred_token = nn.Parameter(torch.normal(0, 0.02, size=(1, 1536)))
        self.gen_token = nn.Parameter(torch.normal(0, 0.02, size=(2, 1536)))
        # Replace the first three layers of QwenAttention with Cross Attention
        assert config.num_cross_layers <= config.num_hidden_layers//2
        mlist = []
        self.insert_idx_vision = [layer_i*5+6 for layer_i in range(config.num_cross_layers)]
        self.insert_idx_audio = [layer_i*5+5 for layer_i in range(config.num_cross_layers)]
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.insert_idx_vision:
                mlist.append(LuckyDecoderLayer(config, layer_idx, "vision"))
            elif layer_idx in self.insert_idx_audio:
                mlist.append(LuckyDecoderLayer(config, layer_idx, "audio"))
            else:
                mlist.append(Qwen2DecoderLayer(config, layer_idx))
        self.layers = nn.ModuleList(mlist)
        self._attn_implementation = config._attn_implementation
        self.norm = Qwen2RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LuckyRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        addi_hidden_states: Optional[torch.FloatTensor] = None,
        img_embeds: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_img: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        mask_vision = None,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You cannot specify both input_ids and inputs_embeds at the same time, and must specify either one"
            )

        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                )
                use_cache = False

        use_legacy_cache = False
        if use_cache and not isinstance(past_key_values, Cache):
            use_legacy_cache = True
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple and this is deprecated and will be removed in v4.43. "
                "Please use an appropriate `Cache` class (https://huggingface.co/docs/transformers/v4.41.3/en/internal/generation_utils#transformers.Cache)"
            )

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
            
            if img_embeds is not None:
                if mask_vision is not None: img_embeds[mask_vision, :] = self.mask_token.to(img_embeds.dtype)
                inputs_embeds[torch.where(input_ids==151652)] = self.special_token[0].to(inputs_embeds.dtype)
                inputs_embeds[torch.where(input_ids==151653)] = self.special_token[1].to(inputs_embeds.dtype)
                inputs_embeds[torch.where(input_ids==151668)] = self.pred_token.to(inputs_embeds.dtype)
                inputs_embeds[torch.where(input_ids==151667)] = self.gen_token[0].to(inputs_embeds.dtype)
                inputs_embeds[torch.where(input_ids==self.img_ids)] = img_embeds.to(inputs_embeds.dtype)

        if self.training:
            #print("test")
            dims = torch.tensor(inputs_embeds.size(1) * inputs_embeds.size(2))
            mag_norm = 6/torch.sqrt(dims)
            inputs_embeds += torch.zeros_like(inputs_embeds).uniform_(-mag_norm, mag_norm)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )
        #print(f"before posi:{position_ids}")
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        #print(f"after posi:{position_ids}")

        causal_mask = self._update_causal_mask(
            attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        )

        ######### need fixed
        #bi-dir self attention
        for i in range(inputs_embeds.shape[0]):
            img_ids_tmp = torch.where(input_ids[i]==self.img_ids)
            causal_mask[i][0][img_ids_tmp] = torch.finfo(causal_mask.dtype).min
            causal_mask[i][0][img_ids_tmp[0][:, None].repeat(1,img_ids_tmp[0].shape[0]).flatten(), img_ids_tmp[0].repeat(img_ids_tmp[0].shape[0])] = 0.
        ######################
        
        mask_type = causal_mask.dtype
        if cross_attention_mask is not None:
            cross_attention_mask = [self._update_cross_mask(mask_type, attention_mask, cross_attention_mask[0]),
                                    [self._update_cross_mask(mask_type, attention_mask, i) for i in cross_attention_mask[1]]]

        hidden_states = inputs_embeds

        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        next_decoder_cache = None

        assert len(addi_hidden_states)==len(self.insert_idx_vision)*2, "vision hidden states and custom layer do not match"
        iter_addi = iter(addi_hidden_states)
        for i, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
                
            if self.gradient_checkpointing and self.training:
                if i in self.insert_idx_vision or i in self.insert_idx_audio:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        next(iter_addi),
                        causal_mask,
                        cross_attention_mask[i%5],
                        position_ids,
                        position_ids_img,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                        mask_vision
                    )
                else:
                    layer_outputs = self._gradient_checkpointing_func(
                        decoder_layer.__call__,
                        hidden_states,
                        causal_mask,
                        position_ids,
                        past_key_values,
                        output_attentions,
                        use_cache,
                        cache_position,
                        position_embeddings,
                    )
            else:
                if i in self.insert_idx_vision or i in self.insert_idx_audio:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        addi_hidden=next(iter_addi),
                        attention_mask=causal_mask,
                        cross_attention_mask=cross_attention_mask[i%5],
                        position_ids=position_ids,
                        position_ids_img=position_ids_img,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                        mask_vision=mask_vision
                    )
                else:
                    layer_outputs = decoder_layer(
                        hidden_states,
                        attention_mask=causal_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_values,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache

        if not return_dict:
            return tuple(v for v in [hidden_states, next_cache, all_hidden_states, all_self_attns] if v is not None)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
    
    # custom cross attention mask
    def _update_cross_mask(self, dtype, attention_mask, cross_attention_mask):
        if cross_attention_mask is None: return None
        device = attention_mask.device
        bsz = attention_mask.shape[0]
        text_query_num = attention_mask.shape[1]
        img_query_num = cross_attention_mask.shape[1]
        mask = torch.full((bsz, text_query_num, img_query_num), torch.finfo(dtype).min).to(device)
        bmm_type = torch.int8 if device==torch.device("cpu") else torch.float16
        mask_cond = torch.bmm(attention_mask.to(bmm_type).unsqueeze(-1),
                              cross_attention_mask.to(bmm_type).unsqueeze(1))
        mask.masked_fill_(mask_cond == 1, 0)
        
        return mask
    
    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
        past_key_values: Cache,
        output_attentions: bool,
    ):
        # TODO: As of torch==2.2.0, the `attention_mask` passed to the model in `generate` is 2D and of dynamic length even when the static
        # KV cache is used. This is an issue for torch.compile which then recaptures cudagraphs at each decode steps due to the dynamic shapes.
        # (`recording cudagraph tree for symint key 13`, etc.), which is VERY slow. A workaround is `@torch.compiler.disable`, but this prevents using
        # `fullgraph=True`. See more context in https://github.com/huggingface/transformers/pull/29114
        device = attention_mask.device.type
        if self.config._attn_implementation == "flash_attention_2":
            print("flash_attention_2")
            if attention_mask is not None and 0.0 in attention_mask:
                return attention_mask
            return None

        # For SDPA, when possible, we will rely on its `is_causal` argument instead of its `attn_mask` argument, in
        # order to dispatch on Flash Attention 2. This feature is not compatible with static cache, as SDPA will fail
        # to infer the attention mask.
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        using_static_cache = isinstance(past_key_values, StaticCache)

        # When output attentions is True, sdpa implementation's forward method calls the eager implementation's forward
        if self.config._attn_implementation == "sdpa" and not using_static_cache and not output_attentions and not device=="cuda":
#             print("attention_mask:", attention_mask.shape)
#             print("input_tensor:", input_tensor.shape)
            if AttentionMaskConverter._ignore_causal_mask_sdpa(
                attention_mask,
                inputs_embeds=input_tensor,
                past_key_values_length=past_seen_tokens,
                is_training=self.training,
            ):
                return None

        dtype, device = input_tensor.dtype, input_tensor.device
        min_dtype = torch.finfo(dtype).min
        sequence_length = input_tensor.shape[1]
        if using_static_cache:
            target_length = past_key_values.get_max_length()
        else:
            target_length = (
                attention_mask.shape[-1]
                if isinstance(attention_mask, torch.Tensor)
                else past_seen_tokens + sequence_length + 1
            )

        if attention_mask is not None and attention_mask.dim() == 4:
            # in this case we assume that the mask comes already in inverted form and requires no inversion or slicing
            if attention_mask.max() != 0:
                raise ValueError("Custom 4D attention mask should be passed in inverted form with max==0`")
            causal_mask = attention_mask
        else:
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(input_tensor.shape[0], 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :]
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )
        if (
            self.config._attn_implementation == "sdpa"
            and attention_mask is not None
            and attention_mask.device.type == "cuda"
            and not output_attentions
        ):
            # Attend to all tokens in fully masked rows in the causal_mask, for example the relevant first rows when
            # using left padding. This is required by F.scaled_dot_product_attention memory-efficient attention path.
            # Details: https://github.com/pytorch/pytorch/issues/110213
            causal_mask = AttentionMaskConverter._unmask_unattended(causal_mask, min_dtype)

        return causal_mask
    
    

@dataclass
class LuckyCausalLMOutputWithPast(CausalLMOutputWithPast):
    addi_hidden_states: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None
    position_ids_img: Optional[torch.Tensor] = None
    
# Copied from transformers.models.idefics3.modeling_idefics3.Idefics3Connector
class VisionConnector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.scale_factor = config.scale_factor
        self.modality_projection = nn.Linear(1152 * (config.scale_factor**2),
                                             config.hidden_size,
                                             bias=False)

    def pixel_shuffle(self, x, scale_factor=3):
        bsz, seq, embed_dim = x.size()
        height = width = int(seq**0.5)
        x = x.view(bsz, height, width, embed_dim)
        x = x.view(bsz, height, int(width / scale_factor), embed_dim * scale_factor)
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(width / scale_factor), int(height / scale_factor), embed_dim * (scale_factor**2))
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(bsz, int(seq / (scale_factor**2)), embed_dim * (scale_factor**2))
        return x

    def forward(self, image_hidden_states):
        image_hidden_states = self.pixel_shuffle(image_hidden_states, self.scale_factor)
        image_hidden_states = self.modality_projection(image_hidden_states)
        return image_hidden_states


class LuckyForCausalLM(Qwen2ForCausalLM):
    config_class = LuckyConfig
    def __init__(self, config):
        super().__init__(config)
        
        print(config.vision_model_name)
        self.model = LuckyModel(config)
        self.vocab_size = config.vocab_size
        try:
            self.patch_size = config.vision_config.patch_size
        except:
            config.vision_config = Idefics2VisionConfig(**config.vision_config)
            self.patch_size = config.vision_config.patch_size
            
        # micro soft swin
        self.vision_tower = self._load_vision_tower_old(config.vision_model_name)

        self.audio_tower = self._load_audio_tower(config.audio_model_name)
        self.audio_dim = config.audio_hidden_size
        self.embed_dim = config.hidden_size
        self.query_num = config.query_num
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.vision_adapter = VisionConnector(config)
        
        #self.audio_adapter = nn.Embedding(16, config.hidden_size)
        #self.audio_adapter = Resampler(
        #    num_queries=config.query_num,
        #    embed_dim=self.embed_dim,
        #    num_heads=self.embed_dim // 128,
        #    kv_dim=self.audio_dim,
        #    adaptive=True,
        #    max_size=(1, 1500)
        #)

        # Diffusion Loss
        self.diffloss = DiffLoss(
           target_channels=16,
           z_channels=1536,
           width=1536,
           depth=12,
           num_sampling_steps="100",
           grad_checkpointing=False
        )
        self.diffusion_pos_embed_learned = nn.Parameter(torch.normal(0, 0.02, size=(1, 24*24, 1536)))
        
        # Initialize weights and apply final processing
        self.post_init()
        
    
    def _load_vision_tower_old(self, vision_model_name):
        model = AutoModel.from_pretrained(vision_model_name)
        model.vision_model.encoder.layers = model.vision_model.encoder.layers[:-1]
        model.vision_model.head = nn.Identity()
        model.requires_grad_(False)
        #model.eval()
        
        return model.vision_model
    
    def _load_vision_tower(self, vision_model_name):
        model = SwinForImageClassification.from_pretrained(vision_model_name)

        model.requires_grad_(False)
        #model.eval()
        
        return model.swin
    
    def _build_adapter(self, hidden_size, vision_dim=1024, num_cross_layers=5):
        mlp = [nn.LayerNorm(vision_dim, elementwise_affine=False),
               nn.GELU(),
               nn.Linear(vision_dim, hidden_size*num_cross_layers)]
        return nn.Sequential(*mlp)
    
    def _load_audio_tower(self, audio_model_name):
        model = WhisperForConditionalGeneration.from_pretrained(audio_model_name, attn_implementation="sdpa")
        model.requires_grad_(False)
        model.eval()
        
        return model.model.encoder
        
    def get_addi_embedding(self, images, audio, vision_tgt_sizes, audio_tgt_sizes):
        dtype = self.vision_tower.embeddings.position_embedding.weight.dtype
        device = self.vision_tower.embeddings.position_embedding.weight.device
        #dtype = self.vision_tower.embeddings.patch_embeddings.projection.weight.dtype
        #device = self.vision_tower.embeddings.patch_embeddings.projection.weight.device
        if images is None:
            vision_embedding = None
        else:
            vision_embeds = []
            vision_embedding_all = []
            for image in images:
                vision_embedding = self.vision_tower(pixel_values=image.type(dtype), output_hidden_states=True)
                #if not (image==0).all():
                vision_embeds.append(self.vision_adapter(vision_embedding.last_hidden_state).squeeze(0))
                    #vision_embeds.append(self.vision_adapter(torch.arange(16, device=device)))
                vision_embedding_all.append(vision_embedding.hidden_states)
                #vision_embedding_all.append(vision_embedding.last_hidden_state.squeeze(0))
            #vision_embedding_all = torch.nn.utils.rnn.pad_sequence(vision_embedding_all, batch_first=True, padding_value=0)
                                     
        if audio is None:
            audio_embedding = None
        else:
            audio_embedding = self.audio_tower(audio.type(dtype), output_hidden_states=True)
            #audio_embedding = self.audio_tower(audio.type(dtype)).last_hidden_state
            #audio_embedding = self.audio_adapter(audio_embedding, audio_tgt_sizes)
        
        
        addi_embedding_list = []
        for i in range(self.config.num_cross_layers):
            addi_embedding_list.append([None, None])
            #addi_embedding_list.append([audio_embedding.hidden_states[-self.config.num_cross_layers+i], audio_tgt_sizes])
            #addi_embedding_list.append([audio_embedding, audio_tgt_sizes])
            tmp_vision_embedding = torch.nn.utils.rnn.pad_sequence([emb[-(i*4+1)].squeeze(0) for emb in vision_embedding_all], batch_first=True, padding_value=0)
            #tmp_vision_embedding = torch.nn.utils.rnn.pad_sequence([emb[(-self.config.num_cross_layers+i)//2].squeeze(0) for emb in vision_embedding_all], batch_first=True, padding_value=0)
            addi_embedding_list.append([tmp_vision_embedding, None])

        if vision_embeds:
            vision_embeds = torch.vstack(vision_embeds)
        else:
            vision_embeds = None

        return addi_embedding_list, vision_embeds
    
    def forward(
        self,
        pixel_values = None,
        audio = None,
        audio_tgt_sizes = None,
        vision_tgt_sizes = None,
        addi_hidden_states: Optional[List[torch.FloatTensor]] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[List[torch.Tensor]] = None,
        cross_attention_mask_img: Optional[torch.Tensor] = None,
        cross_attention_mask_audio: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_: Optional[torch.LongTensor] = None,
        position_ids_img: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        img_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        #cross_attention_mask = [cross_attention_mask_audio, cross_attention_mask_img]
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if addi_hidden_states is None:
            addi_hidden_states, img_embeds = self.get_addi_embedding(pixel_values, audio, vision_tgt_sizes, audio_tgt_sizes)
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            img_embeds=img_embeds,
            addi_hidden_states=addi_hidden_states,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            position_ids=position_ids,
            position_ids_img=position_ids_img,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            #loss_fct = CrossEntropyLoss()
            loss_fct = CrossEntropyLoss(reduction='sum')
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return LuckyCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            addi_hidden_states=addi_hidden_states,
            position_ids_img=position_ids_img,
            position_ids=position_ids,
        )

    def patchify(self, x, patch_size=1):
        bsz, c, h, w = x.shape
        p = patch_size
        h_, w_ = h // p, w // p

        x = x.reshape(bsz, c, h_, p, w_, p)
        x = torch.einsum('nchpwq->nhwcpq', x)
        x = x.reshape(bsz, h_ * w_, c * p ** 2)
        return x  # [n, l, d]

    def unpatchify(self, x, patch_size=1):
        bsz = 2
        p = patch_size
        c = 16
        h_, w_ = 24, 24

        x = x.reshape(bsz, h_, w_, c, p, p)
        x = torch.einsum('nhwcpq->nchpwq', x)
        x = x.reshape(bsz, c, h_ * p, w_ * p)
        return x  # [n, c, h, w]


    def forward_gen(
        self,
        pixel_values = None,
        audio = None,
        audio_tgt_sizes = None,
        vision_tgt_sizes = None,
        zs_tilde = None,
        addi_hidden_states: Optional[List[torch.FloatTensor]] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[List[torch.Tensor]] = None,
        cross_attention_mask_img: Optional[torch.Tensor] = None,
        cross_attention_mask_audio: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_: Optional[torch.LongTensor] = None,
        position_ids_img: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        img_embeds: Optional[torch.FloatTensor] = None, 
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if addi_hidden_states is None:
            addi_hidden_states, img_embeds = self.get_addi_embedding(pixel_values, audio, vision_tgt_sizes, audio_tgt_sizes)
            mask_vision = torch.randint(0,5,size=(img_embeds.shape[0],)).bool()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            img_embeds=img_embeds,
            addi_hidden_states=addi_hidden_states,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            position_ids=position_ids,
            position_ids_img=position_ids_img,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            mask_vision=mask_vision,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states[:, -576:]
        hidden_states = hidden_states + self.diffusion_pos_embed_learned

        
        posterior = DiagonalGaussianDistribution(labels).sample().mul_(0.2325)
        x = self.patchify(posterior)
        gt_latents = x.clone().detach().to(hidden_states.device)
        bsz, seq_len, _ = gt_latents.shape
        gt_latents = gt_latents.reshape(bsz * seq_len, -1).repeat(4, 1)
        hidden_states = hidden_states.reshape(bsz*seq_len, -1).repeat(4, 1)
        zs_tilde = zs_tilde.reshape(bsz*seq_len, -1).repeat(4, 1)
        loss, proj_loss = self.diffloss(z=hidden_states,
                target=gt_latents,mask=None,zs_tilde=zs_tilde)
        return loss, proj_loss

    def sample_tokens(
        self,
        pixel_values = None,
        audio = None,
        audio_tgt_sizes = None,
        vision_tgt_sizes = None,
        zs_tilde = None,
        addi_hidden_states: Optional[List[torch.FloatTensor]] = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        cross_attention_mask: Optional[List[torch.Tensor]] = None,
        cross_attention_mask_img: Optional[torch.Tensor] = None,
        cross_attention_mask_audio: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        position_ids_: Optional[torch.LongTensor] = None,
        position_ids_img: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        img_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        cfg=1.0, temperature=1.0
    ):
        if addi_hidden_states is None:
            addi_hidden_states, img_embeds = self.get_addi_embedding(pixel_values, audio, vision_tgt_sizes, audio_tgt_sizes)
            mask_vision = torch.zeros((img_embeds.shape[0],)).bool()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            img_embeds=img_embeds,
            addi_hidden_states=addi_hidden_states,
            attention_mask=attention_mask,
            cross_attention_mask=cross_attention_mask,
            position_ids=position_ids,
            position_ids_img=position_ids_img,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            mask_vision=mask_vision,
        )

        hidden_states = outputs[0]
        hidden_states = hidden_states[:, -576:]
        hidden_states = hidden_states + self.diffusion_pos_embed_learned

        bsz, seq_len, _ = hidden_states.shape
        hidden_states = hidden_states.reshape(bsz*seq_len, -1)

        sampled_token_latent = self.diffloss.sample(hidden_states, temperature, cfg)
        if not cfg == 1.0:
            sampled_token_latent, _ = sampled_token_latent.chunk(2, dim=0)  # Remove null class samples

        # unpatchify
        sampled_token_latent = self.unpatchify(sampled_token_latent)
        return sampled_token_latent


    def prepare_inputs_for_generation(
        self,
        input_ids,
        data=None,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        use_cache=True,
        **kwargs,
    ):
        position_ids = kwargs.get("position_ids_", None)
        model_inputs = super().prepare_inputs_for_generation(input_ids=input_ids,
                                                             past_key_values=past_key_values,
                                                             attention_mask=attention_mask,
                                                             inputs_embeds=inputs_embeds,
                                                             cache_position=cache_position,
                                                             position_ids=position_ids,
                                                             use_cache=use_cache,
                                                             **kwargs,)
        model_inputs.update(
            {
                "pixel_values": data["pixel_values"],
                "audio_tgt_sizes": data["audio_tgt_sizes"],
                "audio": data["audio"],
                "cross_attention_mask": [data["cross_attention_mask_audio"],
                    data["cross_attention_mask_img"]],
            }
        )
        ids = torch.arange(0,input_ids.shape[1], device=input_ids.device).unsqueeze(0).repeat(input_ids.shape[0], 1)
        model_inputs.update(
            {
                # sigle img need fix
                "position_ids_img": data["position_ids_img"]#ids[torch.where(input_ids==151665)].repeat(3,1).transpose(-2, -1).reshape(input_ids.shape[0], -1).repeat(1,3)
            }
        )
        if "addi_hidden_states" in kwargs.keys():
            model_inputs.update(
            {
                "addi_hidden_states": kwargs["addi_hidden_states"]
            }
            )
            
        if "img_embeds" in kwargs.keys():
            model_inputs.update(
            {
                "img_embeds": kwargs["img_embeds"]
            }
            )
            print(f"img emb shape: {kwargs['img_embeds'].shape}")
            
        device = input_ids.device
        for key, value in model_inputs.items():
            if type(model_inputs[key]) == list:
                if key=="cross_attention_mask":
                    model_inputs[key] = [model_inputs[key][0].to(device),
                                         [data.to(device) for data in model_inputs[key][1]]]
                    continue
                try:
                    model_inputs[key] = [data.to(device) for data in model_inputs[key]]
                except:
                    continue
            elif model_inputs[key] is None or type(model_inputs[key]) == bool:
                continue
            else:
                model_inputs[key] = model_inputs[key].to(device)
        return model_inputs
    
    def _update_model_kwargs_for_generation(self, outputs, model_kwargs, is_encoder_decoder, **kwargs):
        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs=outputs,
            model_kwargs=model_kwargs,
            is_encoder_decoder=is_encoder_decoder,
            **kwargs,
        )
        # Get the precomputed image_hidden_states
        model_kwargs["addi_hidden_states"] = outputs.addi_hidden_states
        model_kwargs["position_ids_img"] = outputs.position_ids_img
        max_id_add = outputs.position_ids.max(dim=-1).values.max(0).values + 1 # max ids / batch
        model_kwargs["position_ids_"] = max_id_add.repeat(1,2,1).permute(1, 2, 0) # 2, bs, n
        #model_kwargs["img_embeds"] = outputs.img_embeds
        return model_kwargs
    
    def _get_train_param(self, stage=1):
        param_list = []
        for name, param in self.named_parameters():
            if "_adapter" in name and stage in [1,2,3] and any(f".{str(layer_ind)}." in name for layer_ind in [6,11,16,21,26]):
                print(name)
                param.requires_grad = True
                param_list.append(param)
            elif "proj_addi" in name and stage in [1,2,3] and any(f".{str(layer_ind)}." in name for layer_ind in [6,11,16,21,26]):
                print(name)
                param.requires_grad = True
                param_list.append(param)
            elif "lora" in name and stage in [3]:
                print(name)
                param.requires_grad = True
                param_list.append(param)
            elif "B" in name and stage in [5, 6]:
                print(name)
                param.requires_grad = True
                param_list.append(param)
            elif "vision_adapter" in name and stage in [1,2,3]:
                print(name)
                param.requires_grad = True
                param_list.append(param)
            #elif "embed_tokens" in name and stage==3:
            #    print(name)
            #    param.requires_grad = True
            #    param_list.append(param)
            elif "self_attn.gate" in name and stage in [1,2,3] and any(f".{str(layer_ind)}." in name for layer_ind in [6,11,16,21,26]):
                print(name)
                param.requires_grad = True
                param_list.append(param)
            elif "vision_tower" in name and stage==2:
               print(name)
               param.requires_grad = True
               param_list.append(param)
            # elif "special_token" in name and stage in [1,2,3]:
            #    print(name)
            #    param.requires_grad = True
            #    param_list.append(param)
            elif "gen_token" in name and stage in [4, 6]:
               print(name)
               param.requires_grad = True
               param_list.append(param)
            elif "pred_token" in name and stage in [4, 6]:
               print(name)
               param.requires_grad = True
               param_list.append(param)
            elif "mask_token" in name and stage in [4, 6]:
               print(name)
               param.requires_grad = True
               param_list.append(param)
            elif "diffusion_pos_embed_learned" in name and stage in [4, 6]:
               print(name)
               param.requires_grad = True
               param_list.append(param)
            elif "diffloss" in name and stage in [4, 6]:
               print(name)
               param.requires_grad = True
               param_list.append(param)
            #elif "mlp" in name:
            #    param.requires_grad = False
            #elif any(f".{str(layer_ind)}." in name for layer_ind in range(0, 28)) and "model" in name:
            #    print(name)
            #    param.requires_grad = True
            #    param_list.append(param)
            else:
                param.requires_grad = False
               
        return param_list
    
    

