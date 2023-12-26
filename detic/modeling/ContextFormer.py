import copy
from typing import Optional, List
import math
import torch
import torch.nn.functional as F
from torch import nn, Tensor
import numpy as np
from timm.models.layers import trunc_normal_
# from detic.modeling.sam.modeling.prompt_encoder import PositionEmbeddingRandom
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        """
        tgt: [bsz, tgt_len, dim]
        """
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, tgt_mask, memory_mask,
                                    tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, tgt_mask, memory_mask,
                                 tgt_key_padding_mask, memory_key_padding_mask, pos, query_pos)

class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        # postion_embedding should be same to image pathes + 1 


    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(output, memory, tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           pos=pos, query_pos=query_pos)
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output

class build_contextformer(nn.Module):  
    def __init__(self,
                mask_dim=1024,
                d_model=256,
                clip_txt_dim=512,
                nhead=8,
                num_decoder_layers=3,
                normalize_before=False,
                dim_feedforward=2048,
                dropout=0.1,
                activation="relu",
                return_intermediate_dec=False,
                use_ln=True) -> None:
        """
        use_ln: whether use ln to visual tokens and masktokens(layernorm)
        """
        super().__init__()
        attenLayer1 = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        attenLayer2 = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)

        decoder_norm1 = nn.LayerNorm(d_model)
        decoder_norm2 = nn.LayerNorm(d_model)
        self.decoder1 = TransformerDecoder(attenLayer1, num_decoder_layers, decoder_norm1,
                                            return_intermediate=return_intermediate_dec)
        self.decoder2 = TransformerDecoder(attenLayer2, num_decoder_layers, decoder_norm2,
                                            return_intermediate=return_intermediate_dec)
        self.q_proj = nn.Linear(mask_dim, d_model) # for mask_token
        self.kv_proj = nn.Linear(2*clip_txt_dim, d_model) # for k,v
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.pe = PositionalEncoding(d_model)

    def get_qs(self, q, cls):
        # concat[cls*txt,txt]
        bs, _ = cls.shape
        q = q.expand(bs, -1, -1)

        q1 = torch.einsum("bd,bcd->bcd", cls, q) #bs, dim, C
        q_ = torch.concat((q1, q),dim=-1) # for cls token and text token have align, there are no laynorms
        return q_

    def forward(self, mask_token, clip_vis, clip_txt):
        # for clip res50, cls_token not exits
        cls_token, visual_tokens = clip_vis[:,0], clip_vis[:,1:]
        cls_token = self.get_qs(clip_txt, cls_token)
        kv = self.kv_proj(cls_token)

        if self.use_ln:
            q = self.ln_mask(self.q_proj(mask_token))
        else:
            q = self.q_proj(mask_token)
        mask_text = self.decoder1(q, kv)
        pe = self.pe(visual_tokens.shape[-2]).unsqueeze(0)
        pe = torch.repeat_interleave(pe, visual_tokens.shape[0], dim=0)
        mask_img = self.decoder2(mask_text, visual_tokens, pos=pe)
        return self.get_logits(mask_img.squeeze(), clip_txt)

    def get_logits(self, image, text):
        image = image/(image.norm(dim=-1, keepdim=True) + 1e-7)
        text = text/(text.norm(dim=-1, keepdim=True)+ 1e-7)
    
        logit_scale = self.logit_scale.exp()
        mask_cls_img = logit_scale * image @ text.t()
        mask_cls_txt = mask_cls_img.t()
        return mask_cls_img, mask_cls_txt
    
class build_yhs_contextFormer(nn.Module):
    def __init__(self,
        mask_dim=1024,
        d_model=256,
        vis_dim=2048,
        nhead=8,
        num_decoder_layers=3,
        normalize_before=True,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        return_intermediate_dec=False) -> None:
        """
        use_ln: whether use ln to visual tokens and masktokens(layernorm)
        """
        super().__init__()
        # defalut: set d_model = clip_txt_dim
        attenLayer1= TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm1 = nn.LayerNorm(d_model)
        self.decoder1 = TransformerDecoder(attenLayer1, num_decoder_layers, decoder_norm1,
                                            return_intermediate=return_intermediate_dec)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.linear = nn.Linear(mask_dim, d_model)
        self.vis_linear = nn.Linear(vis_dim, d_model)

    def forward(self, mask_token, clip_vis, clip_txt, pos, query_pos):
        # for clip res50, cls_token not exits
        clip_vis = clip_vis.flatten(start_dim=2)
        clip_vis = clip_vis.permute(0,2,1)
        clip_vis = self.vis_linear(clip_vis)
        mask_token = self.linear(mask_token)
        semantic_token = self.decoder1(mask_token, clip_vis, pos, query_pos)
        return self.get_logits(semantic_token, clip_txt)

    def get_logits(self, image, text):
        image = image/(image.norm(dim=-1, keepdim=True)+ 1e-7)
        text = text/(text.norm(dim=-1, keepdim=True)+ 1e-7)
        logit_scale = self.logit_scale.exp()
        mask_cls_img = logit_scale * image @ (text.t())
        return mask_cls_img

class PositionalEncoding(nn.Module):
    def __init__(self, D, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        # Create a long enough `pe` matrix
        pe = torch.zeros(max_len, D)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, D, 2).float() * -(math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register `pe` as a buffer
        self.register_buffer('pe', pe)

    def forward(self, size):
        # Add `pe` to input
        return self.pe[:size, :]