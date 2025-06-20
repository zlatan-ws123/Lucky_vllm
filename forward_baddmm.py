import torch
from torch import nn
from transformers.cache_utils import Cache, DynamicCache, EncoderDecoderCache, StaticCache
from typing import List, Optional, Tuple, Union
from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb
from torch.nn.attention import SDPBackend, sdpa_kernel


def forward_qwen_baddmm(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.46
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
                                 query_states.view(-1, q_len, self.head_dim),
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

    return attn_output, attn_weights, past_key_value


def forward_whisper_baddmm(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[EncoderDecoderCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

    if past_key_value is not None:
        is_updated = past_key_value.is_updated.get(self.layer_idx)
        if is_cross_attention:
            # after the first generated id, we can subsequently re-use all key/value_states from cache
            past_key_value.is_updated[self.layer_idx] = True
            past_key_value = past_key_value.cross_attention_cache
        else:
            past_key_value = past_key_value.self_attention_cache

    # use key_value_states if cross attention
    current_states = key_value_states if key_value_states is not None else hidden_states
    if is_cross_attention and past_key_value and is_updated:
        # reuse k,v, cross_attentions
        key_states = past_key_value.key_cache[self.layer_idx]
        value_states = past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self._shape(self.k_proj(current_states), -1, bsz)
        value_states = self._shape(self.v_proj(current_states), -1, bsz)
        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

    # Preallocate attn_weights for `baddbmm`(baddbmm faster)
    attn_weights = torch.empty(
        bsz * self.num_heads, tgt_len, key_states.shape[-2], dtype=query_states.dtype, device=query_states.device
    )
    attn_weights = torch.baddbmm(attn_weights,
                                 query_states.view(-1, tgt_len, self.head_dim),
                                 key_states.transpose(2, 3).view(-1, self.head_dim, key_states.shape[-2]),
                                 beta=0, alpha=self.scaling
                                 ).view(bsz, self.num_heads, tgt_len, key_states.shape[-2])

    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)

    if layer_head_mask is not None:
        if layer_head_mask.size() != (self.num_heads,):
            raise ValueError(
                f"Head mask for a single layer should be of size {(self.num_heads,)}, but is"
                f" {layer_head_mask.size()}"
            )
        attn_weights = layer_head_mask.view(1, -1, 1, 1) * attn_weights

    attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)
    attn_output = torch.einsum("bhts,bhsd->bthd", attn_weights, value_states)

    if attn_output.size() != (bsz, tgt_len, self.num_heads, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, tgt_len, self.num_heads, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)
    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, attn_weights, past_key_value

def forward_whisper_sdpa(
    self,
    hidden_states: torch.Tensor,
    key_value_states: Optional[torch.Tensor] = None,
    past_key_value: Optional[EncoderDecoderCache] = None,
    attention_mask: Optional[torch.Tensor] = None,
    layer_head_mask: Optional[torch.Tensor] = None,
    output_attentions: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """Input shape: Batch x Time x Channel"""
    if output_attentions or layer_head_mask is not None:
        # TODO: Improve this warning with e.g. `model.config._attn_implementation = "manual"` once this is implemented.
        logger.warning_once(
            "WhisperModel is using WhisperSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True` or `layer_head_mask` not None. Falling back to the manual attention"
            ' implementation, but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states,
            key_value_states=key_value_states,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
            cache_position=cache_position,
        )

    # if key_value_states are provided this layer is used as a cross-attention layer
    # for the decoder
    is_cross_attention = key_value_states is not None
    bsz, tgt_len, _ = hidden_states.size()

    # get query proj
    query_states = self._shape(self.q_proj(hidden_states), tgt_len, bsz)

    if past_key_value is not None:
        is_updated = past_key_value.is_updated.get(self.layer_idx)
        if is_cross_attention:
            # after the first generated id, we can subsequently re-use all key/value_states from cache
            past_key_value.is_updated[self.layer_idx] = True
            past_key_value = past_key_value.cross_attention_cache
        else:
            past_key_value = past_key_value.self_attention_cache

    # use key_value_states if cross attention
    current_states = key_value_states if key_value_states is not None else hidden_states
    if is_cross_attention and past_key_value and is_updated:
        # reuse k,v, cross_attentions
        key_states = past_key_value.key_cache[self.layer_idx]
        value_states = past_key_value.value_cache[self.layer_idx]
    else:
        key_states = self._shape(self.k_proj(current_states), -1, bsz)
        value_states = self._shape(self.v_proj(current_states), -1, bsz)
        if past_key_value is not None:
            # save all key/value_states to cache to be re-used for fast auto-regressive generation
            cache_position = cache_position if not is_cross_attention else None
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, {"cache_position": cache_position}
            )

    causal_mask = attention_mask
    if attention_mask is not None:  # no matter the length, we just slice it
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this `is_causal` if statement instead of an inline conditional assignment
    # in SDPA to support both torch.compile's dynamic shapes and full graph options. An inline conditional prevents dynamic shapes from compiling.
    # The tgt_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case tgt_len == 1.
    is_causal = True if self.is_causal and causal_mask is None and tgt_len > 1 else False

    # NOTE: SDPA with memory-efficient backend is currently (torch==2.1.2) bugged when using non-contiguous inputs and a custom attn_mask,
    # but we are fine here as `_shape` do call `.contiguous()`. Reference: https://github.com/pytorch/pytorch/issues/112577
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        attn_output = torch.nn.functional.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=is_causal,
        )

    if attn_output.size() != (bsz, self.num_heads, tgt_len, self.head_dim):
        raise ValueError(
            f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
            f" {attn_output.size()}"
        )

    attn_output = attn_output.transpose(1, 2)

    # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
    # partitioned across GPUs when using tensor-parallelism.
    attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

    attn_output = self.out_proj(attn_output)

    return attn_output, None, past_key_value
