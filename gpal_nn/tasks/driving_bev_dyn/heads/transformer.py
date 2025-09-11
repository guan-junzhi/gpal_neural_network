import torch.nn as nn
import torch
import torch.nn.functional as F
import copy
import math
import numpy as np
from typing import Optional, List
from torch import nn, Tensor
from .multihead_attention import MultiheadAttention


class InstanceL2Norm(nn.Module):
    def __init__(self, size_average=True, eps=1e-5, scale=1.0):
        super().__init__()

    def forward(self, p):
        p_mean = p.mean(dim=2).view(1,64,1)
        p_std = p.std(dim=2,unbiased=False).view(1,64,1)
        p = (p-p_mean)/(p_std+0.00001)
        return p


class TransformerEncoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        self.norm = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed
    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, src, query_pos=None):
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(src, query_pos_embed)

        src2 = self.self_attn(query=query, key=key, value=value)
        src = src + src2

        src = self.norm(src.permute(1, 2, 0)).permute(2, 0, 1)
        return F.relu(src)


class TransformerEncoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_encoder_layers=6,
                 activation="relu",
                 self_posembed=None):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(
            multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(encoder_layer, num_encoder_layers)

    def forward(self, src, query_pos=None):
        num_imgs, batch, dim = src.shape
        output = src
        for layer in self.layers:
            output = layer(output, query_pos=query_pos)
        return output


class TransformerDecoderLayer(nn.Module):
    def __init__(self, multihead_attn, FFN, d_model, self_posembed=None):
        super().__init__()
        self.self_attn = multihead_attn
        self.cross_attn = MultiheadAttention(
            feature_dim=d_model,
            n_head=1, key_feature_dim=128)

        self.norm1 = nn.InstanceNorm1d(d_model)
        self.norm2 = nn.InstanceNorm1d(d_model)
        self.self_posembed = self_posembed

    def with_pos_embed(self, tensor, pos_embed: Optional[Tensor]):
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(self, tgt, memory, query_pos=None,track=None):
        if self.self_posembed is not None and query_pos is not None:
            query_pos_embed = self.self_posembed(query_pos).permute(2, 0, 1)
        else:
            query_pos_embed = None
        query = key = value = self.with_pos_embed(tgt, query_pos_embed)
        tgt2 = self.self_attn(query=query, key=key, value=value)
        tgt = tgt + tgt2
        tgt = self.norm1(tgt.permute(1, 2, 0)).permute(2, 0, 1)
        tgt = F.relu(tgt)
        mask,track_mask,mask1 = self.cross_attn(
            query=tgt, key=memory, value=memory,track=True)
        mem = memory + mask1
        mem = self.norm2(mem.permute(1, 2, 0)).permute(2, 0, 1)
        mem = F.relu(mem)
        tgt2 = tgt + mask
        tgt2 = self.norm2(tgt2.permute(1, 2, 0)).permute(2, 0, 1)
        tgt2 = F.relu(tgt2)
        return tgt2,track_mask,mem


class TransformerDecoder(nn.Module):
    def __init__(self, multihead_attn, FFN,
                 d_model=512,
                 num_decoder_layers=6,
                 activation="relu",
                 self_posembed=None):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(
            multihead_attn, FFN, d_model, self_posembed=self_posembed)
        self.layers = _get_clones(decoder_layer, num_decoder_layers)

    def forward(self, tgt, memory, query_pos=None,track=None):
        assert tgt.dim() == 3, 'Expect 3 dimensional inputs'
        tgt_shape = tgt.shape
        num_imgs, batch, dim = tgt.shape

        output = tgt
        for layer in self.layers:
            output,track_mask,output1 = layer(output, memory, query_pos=query_pos,track=track)
        return output,track_mask,output1

def _get_clones(module, N):
    return nn.ModuleList([module for i in range(N)])

def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")