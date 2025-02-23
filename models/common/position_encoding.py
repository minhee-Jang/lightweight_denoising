# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
From DETR and various other models
"""
import math
import torch
from torch import nn
from torch import Tensor

from utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    https://github.com/facebookresearch/detr
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        dim_t = self.temperature ** (2 * torch.div(dim_t, 2, rounding_mode='trunc') / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos

class PositionEmbeddingSineTokens(nn.Module):
    """
    https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    """
    def __init__(self, num_pos_feats: int, dropout: float = 0.1, maxlen: int = 100, n_token: int =0):
        super().__init__()

        if num_pos_feats % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(num_pos_feats))

        length = maxlen + n_token
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, num_pos_feats, 2, dtype=torch.float) *
                            -(math.log(10000.0) / num_pos_feats)))

        pe = torch.zeros(length, num_pos_feats)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)

        if n_token > 0:
            self.embedding = nn.Embedding(n_token, num_pos_feats)

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, num_pos_feats))
        self.n_token = n_token
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        n, bs, emb_sz = x.shape
        x = x.flatten(2)
        
        # print('flatten x.shape:', x.shape)
        # x = x.transpose(-1, -2)
        # print('x.shape:', x.shape)
        # print('self.pe:', self.pe.shape)

        if self.n_token > 0:
            cls_tokens = []
            for i in range(self.n_token):
                # print('self.cls_token.expand(bs, -1, -1).shape:', self.cls_token.expand(n, -1, -1).shape)
                cls_idx = torch.tensor([i], dtype=torch.long, device=torch.device(x.device))
                # print(cls_idx)
                cls_emb = self.embedding(cls_idx)
                # print('cls_emb.shape:', cls_emb.shape)
                cls_emb = cls_emb.expand(1, bs, emb_sz)
                # print('expand cls_emb.shape:', cls_emb.shape)
                # cls_tokens = [self.cls_token.expand(n, -1, -1) for _ in range(self.n_token)]
                cls_tokens.append(cls_emb)
                
            x = torch.cat((*cls_tokens, x), dim=0)


        # print('cat x.shape:', x.shape)

        # print('pe.shape:', self.pe.shape)
        emb = x + self.pe[:x.shape[0]]
        # print('emb.shape:', emb.shape)

        return self.dropout(emb)

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding
