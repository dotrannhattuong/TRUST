import copy

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

class PatchEmbed(nn.Module):
    def __init__(self, img_size=256, patch_size=8, in_chans=3, embed_dim=512):
        super().__init__()
        image_height = image_width = img_size
        patch_height = patch_width = patch_size
        self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)

        return x

class SourceEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=8, d_model=512, nhead=8, num_encoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        # Patch embedding module.
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size)

        # Learnable positional embedding of shape [num_patches, 1, d_model]
        self.pos_embedding = nn.Parameter(torch.randn(self.patch_embed.num_patches, 1, d_model))

        # Multiple encoder layers
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder_c = _get_clones(encoder_layer, num_encoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_img):
        ### Patch embedding ###
        source_embeds = self.patch_embed(source_img)
        source_embeds = source_embeds.flatten(2).permute(2, 0, 1)

        ### Feature extraction ###
        source_feats = source_embeds
        for layer in self.encoder_c:
            source_feats = layer(source_feats, pos=self.pos_embedding)

        return source_feats

class TargetEncoder(nn.Module):
    def __init__(self, img_size=256, patch_size=8, d_model=512, nhead=8, num_encoder_layers=3,
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size)
        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.encoder_s = _get_clones(encoder_layer, num_encoder_layers)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, target_img):
        ### Patch embedding ###
        target_embeds = self.patch_embed(target_img)
        target_embeds = target_embeds.flatten(2).permute(2, 0, 1)

        ### Feature extraction ###
        target_feats = target_embeds
        for layer in self.encoder_s:
            target_feats = layer(target_feats, pos=None)
        
        return target_feats

class TokenDriven(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_decoder_layers=3, 
                 dim_feedforward=2048, dropout=0.1, activation="relu"):
        super().__init__()
        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        self.decoder = _get_clones(decoder_layer, num_decoder_layers)
        self.norm = nn.LayerNorm(d_model)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, source_feats, target_feats, pos=None):
        ### Fusion of source and target features ###
        hs = source_feats
        for layer in self.decoder:
            hs = layer(hs, target_feats, pos=pos)

        if self.norm is not None:
            hs = self.norm(hs)

        ### HWxNxC to NxCxHxW to
        N, B, C = hs.shape          
        H = int(np.sqrt(N))
        hs = hs.permute(1, 2, 0)
        hs = hs.view(B, C, -1, H)

        return hs

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, src, pos=None):
        q = k = self.with_pos_embed(src, pos)

        src2 = self.self_attn(query=q, key=k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu"):
        super().__init__()
        # d_model embedding dim
        self.self_attn_1 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.self_attn_2 = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
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

        self.d_model = d_model

    def with_pos_embed(self, tensor, pos=None):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos=None):
        q = self.with_pos_embed(tgt, pos)
        k = v = memory
 
        tgt2 = self.self_attn_1(q, k, v)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        tgt2 = self.self_attn_2(query=self.with_pos_embed(tgt, pos),
                                key=k, value=v)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")
