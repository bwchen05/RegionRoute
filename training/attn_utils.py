import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple

from diffusers.models.transformers.transformer_flux import FluxAttention
from diffusers.models.transformers.transformer_flux import _get_qkv_projections, _get_fused_projections, _get_projections, dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb

class TrainingAttentionStore:

    def __init__(self):
        self.selective_attention_maps = {}
        self.target_token_indices = []
        self.target_img_len = None
        self.text_len = None

    def set_current_params(self, target_img_len, text_len):
        self.target_img_len = target_img_len
        self.text_len = text_len

    def get_current_params(self):
        return self.target_img_len, self.text_len

    def set_target_tokens(self, token_indices):
        self.target_token_indices = list(set(token_indices))

    def store_selective_attention(self, attention_scores, layer_name):
        self.selective_attention_maps[layer_name] = attention_scores

    def get_attention_maps(self):
        return self.selective_attention_maps

    def reset(self):
        self.selective_attention_maps.clear()
        self.target_token_indices = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

class FluxAttnProcessorForTraining:
    _attention_backend = None

    def __init__(self, attention_store: TrainingAttentionStore = None, layer_name: str = "unknown"):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(f"{self.__class__.__name__} requires PyTorch 2.0.")

        self.attention_store = attention_store
        self.layer_name = layer_name

    def __call__(
        self,
        attn: "FluxAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        image_rotary_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        if self.attention_store is not None:
            target_img_len, text_len = self.attention_store.get_current_params()

        query, key, value, encoder_query, encoder_key, encoder_value = _get_qkv_projections(
            attn, hidden_states, encoder_hidden_states
        )

        query = query.unflatten(-1, (attn.heads, -1))
        key = key.unflatten(-1, (attn.heads, -1))
        value = value.unflatten(-1, (attn.heads, -1))

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        if attn.added_kv_proj_dim is not None:
            encoder_query = encoder_query.unflatten(-1, (attn.heads, -1))
            encoder_key = encoder_key.unflatten(-1, (attn.heads, -1))
            encoder_value = encoder_value.unflatten(-1, (attn.heads, -1))

            encoder_query = attn.norm_added_q(encoder_query)
            encoder_key = attn.norm_added_k(encoder_key)

            query = torch.cat([encoder_query, query], dim=1)
            key = torch.cat([encoder_key, key], dim=1)
            value = torch.cat([encoder_value, value], dim=1)

        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)

        batch_size, seq_len, num_heads, head_dim = query.shape
        query_for_attn = query.transpose(1, 2)
        key_for_attn = key.transpose(1, 2)

        hidden_states = dispatch_attention_fn(
            query, key, value, attn_mask=attention_mask, backend=self._attention_backend
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            if (self.attention_store.target_token_indices and
                encoder_hidden_states is not None):

                txt_len = encoder_hidden_states.shape[1]
                target_indices = self.attention_store.target_token_indices
                valid_indices = [idx for idx in target_indices if idx < txt_len]

                if valid_indices and target_img_len and txt_len + target_img_len <= seq_len:
                    q_img = query_for_attn[:, :, txt_len:txt_len+target_img_len, :]
                    k_txt_selected = key_for_attn[:, :, valid_indices, :]

                    scale = head_dim ** -0.5
                    partial_scores = torch.matmul(q_img, k_txt_selected.transpose(-2, -1)) * scale
                    partial_weights = F.softmax(partial_scores, dim=-1)

                    avg_weights = partial_weights.mean(dim=1)
                    self.attention_store.store_selective_attention(avg_weights, self.layer_name)

            encoder_hidden_states, hidden_states = hidden_states.split_with_sizes(
                [encoder_hidden_states.shape[1], hidden_states.shape[1] - encoder_hidden_states.shape[1]], dim=1
            )
            hidden_states = attn.to_out[0](hidden_states)
            hidden_states = attn.to_out[1](hidden_states)
            encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if (self.attention_store is not None and
                self.attention_store.target_token_indices and
                text_len and target_img_len):

                txt_len = int(text_len)
                target_indices = self.attention_store.target_token_indices
                valid_indices = [idx for idx in target_indices if idx < txt_len]

                if valid_indices and txt_len + target_img_len <= seq_len:
                    q_img = query_for_attn[:, :, txt_len:txt_len+int(target_img_len), :]
                    k_txt_selected = key_for_attn[:, :, valid_indices, :]

                    scale = head_dim ** -0.5
                    partial_scores = torch.matmul(q_img, k_txt_selected.transpose(-2, -1)) * scale
                    partial_weights = F.softmax(partial_scores, dim=-1)
                    avg_weights = partial_weights.mean(dim=1)

                    self.attention_store.store_selective_attention(avg_weights, self.layer_name)
            return hidden_states

def register_flux_attention_for_training(flux_model, attention_store: TrainingAttentionStore):

    total_layers = 0
    attn_procs = {}

    current_processors = flux_model.attn_processors

    for name, current_processor in current_processors.items():
        if 'transformer_blocks' in name:
            attn_procs[name] = FluxAttnProcessorForTraining(attention_store, name)
            total_layers += 1
        else:
            attn_procs[name] = current_processor

    flux_model.set_attn_processor(attn_procs)
    print(f"Registered {total_layers} attention layers for training")

