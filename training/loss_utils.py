import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from copy import deepcopy

def get_word_idx(text: str, tgt_word, tokenizer):

    tgt_word = tgt_word.lower()
    text = text.lower()

    encoded_text = tokenizer.encode(text)[:-1]
    encoded_tgt_word = tokenizer.encode(tgt_word)[:-1]

    first_token_idx = -1
    for i in range(len(encoded_text)):
        if encoded_text[i] == encoded_tgt_word[0]:
            if len(encoded_tgt_word) == 1:
                first_token_idx = i
                break

            following_match = True
            for j in range(1, len(encoded_tgt_word)):
                if i + j >= len(encoded_text) or encoded_text[i + j] != encoded_tgt_word[j]:
                    following_match = False
                    break
            if following_match:
                first_token_idx = i
                break
    if first_token_idx == -1:
        raise ValueError(f"Target word '{tgt_word}' not found in text '{text}'")

    tgt_word_tokens_idx_ls = [i + first_token_idx for i in range(len(encoded_tgt_word))]

    return tgt_word_tokens_idx_ls

def cal_attn_loss_by_layer(attention_store, gt_seg_list, word_token_idx_ls, latent_size, tau=1.0, alpha=1.0):
    selective_attention_maps = attention_store.selective_attention_maps
    target_token_indices = attention_store.target_token_indices

    if not selective_attention_maps or not target_token_indices:
        print("Warning: No selective attention maps found in attention store")
        zero_loss = torch.tensor(0.0, requires_grad=True)
        return {"focus_loss": zero_loss, "cover_loss": zero_loss}

    H, W = latent_size
    device = next(iter(selective_attention_maps.values())).device

    processed_masks = []
    for gt_seg in gt_seg_list:
        if gt_seg.dim() == 4:
            gt_seg = gt_seg.squeeze(0).squeeze(0)
        elif gt_seg.dim() == 3:
            gt_seg = gt_seg.squeeze(0)

        if gt_seg.shape != (H, W):
            gt_seg = F.interpolate(
                gt_seg.unsqueeze(0).unsqueeze(0),
                size=(H, W),
                mode='nearest'
            ).squeeze()

        gt_seg = (gt_seg > 0.0).float()
        processed_masks.append(gt_seg)

    total_focus_loss = 0.0
    total_cover_loss = 0.0
    num_layers = 0

    for layer_name, selective_attention in selective_attention_maps.items():
        B, target_img_len, num_target_tokens = selective_attention.shape

        if target_img_len == H * W:
            reshaped_attention = selective_attention.view(B, H, W, num_target_tokens)
        else:
            reshaped_attention = F.interpolate(
                selective_attention.permute(0, 2, 1),
                size=H * W,
                mode='linear',
                align_corners=False
            ).permute(0, 2, 1).view(B, H, W, num_target_tokens)

        layer_focus_loss = 0.0
        layer_cover_loss = 0.0
        num_valid_objects = 0

        for obj_idx, word_token_indices in enumerate(word_token_idx_ls):
            if obj_idx >= len(processed_masks):
                continue

            if not word_token_indices:
                continue

            mask = processed_masks[obj_idx].to(device)

            obj_focus_loss = 0.0
            obj_cover_loss = 0.0
            num_valid_tokens = 0

            for token_idx in word_token_indices:
                if token_idx not in target_token_indices:
                    continue

                local_idx = target_token_indices.index(token_idx)
                token_attention = reshaped_attention[:, :, :, local_idx]

                pred_attn_flat = token_attention.view(B, -1)
                mask_flat = mask.view(1, -1).expand(B, -1)

                token_focus_loss = 0.0
                for b in range(B):
                    pred_softmax = F.softmax(pred_attn_flat[b] / tau, dim=0)

                    mask_sum = mask_flat[b].sum()
                    if mask_sum > 0:
                        mask_normalized = mask_flat[b] / mask_sum
                    else:
                        mask_normalized = torch.ones_like(mask_flat[b]) / mask_flat[b].numel()

                    kl_div = F.kl_div(
                        pred_softmax.log(),
                        mask_normalized,
                        reduction='sum'
                    )
                    token_focus_loss += kl_div

                token_focus_loss = token_focus_loss / B
                obj_focus_loss += token_focus_loss

                scaled_pred = alpha * pred_attn_flat

                bce_per_sample = F.binary_cross_entropy_with_logits(
                    scaled_pred,
                    mask_flat,
                    reduction='none'
                ).mean(dim=1)

                token_cover_loss = bce_per_sample.mean()

                obj_cover_loss += token_cover_loss
                num_valid_tokens += 1

            if num_valid_tokens > 0:
                layer_focus_loss += obj_focus_loss / num_valid_tokens
                layer_cover_loss += obj_cover_loss / num_valid_tokens
                num_valid_objects += 1

        if num_valid_objects > 0:
            layer_focus_loss = layer_focus_loss / num_valid_objects
            layer_cover_loss = layer_cover_loss / num_valid_objects

            total_focus_loss += layer_focus_loss
            total_cover_loss += layer_cover_loss
            num_layers += 1

    if num_layers > 0:
        total_focus_loss = total_focus_loss / num_layers
        total_cover_loss = total_cover_loss / num_layers
    else:
        total_focus_loss = torch.tensor(0.0, device=device, requires_grad=True)
        total_cover_loss = torch.tensor(0.0, device=device, requires_grad=True)

    return {
        "focus_loss": total_focus_loss,
        "cover_loss": total_cover_loss,
    }

