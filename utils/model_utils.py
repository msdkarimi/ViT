import torch
import numpy as np
import torch.nn as nn


class PatchEmbedding(nn.Module):
    def __init__(self, *, img_dim, input_channel, patch_dim):
        super(PatchEmbedding, self).__init__()
        height, width = img_dim
        assert height % patch_dim == 0, "input image's height dimension must be dividable buy patch_dim"
        assert width % patch_dim == 0, "input image's width dimension must be dividable buy patch_dim"

        emb_dim = input_channel * patch_dim ** 2
        num_patches = (img_dim[0] // patch_dim) * (img_dim[1] // patch_dim)
        self.perform_patching_on_input = nn.Conv2d(input_channel, emb_dim, kernel_size=patch_dim, stride=patch_dim)  # b_s, emb_dim, 14, 14
        self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim), requires_grad=True)
        self.positional_encoding = nn.Parameter(torch.zeros(num_patches + 1, emb_dim)).unsqueeze(0)

    @classmethod
    def from_config(cls, config):
        return cls(img_dim=(config['INPUT_H'], config['INPUT_W']), input_channel=config['INPUT_CHANNEL'], patch_dim=config['PATCH_DIM'])

    def forward(self, x):
        x = self.perform_patching_on_input(x).flatten(2).transpose(1, 2)  # b_s, num_patches, emb_dim
        b_s, _, _ = x.shape
        cls_tokes = self.cls_token.expand(b_s, -1, -1)
        x = torch.cat((cls_tokes, x), dim=1)
        x += self.positional_encoding
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, *, emb_dim: int, dropout: float, head: int, qkv_bias: bool):
        super(MultiHeadSelfAttention, self).__init__()

        assert emb_dim % head == 0, 'it is not dividable by number of heads'

        self.emb_dim = emb_dim
        self.head = head
        self.emb_dim_head = emb_dim // head

        self.wq = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.wk = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.wv = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.output = nn.Linear(emb_dim, emb_dim, bias=qkv_bias)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(p=dropout)

    @classmethod
    def from_config(cls, config, *, emb_dim: int):
        return cls(emb_dim=emb_dim, dropout=config['DROP_OUT'], head=config['HEADS'], qkv_bias=config['HEAD_BIAS'])

    def __create_heads(self, input_tensor):
        return input_tensor.view(input_tensor.shape[0], input_tensor.shape[1], self.head, self.emb_dim_head).transpose(1, 2)

    def __concatenate_heads(self, output):
        transposed_tensor = output.transpose(1, 2).contiguous()
        return transposed_tensor.view(transposed_tensor.shape[0], -1, self.emb_dim)

    def __scaled_dot_product_attention(self, qs, ks, vs):
        d_k = float(ks.shape[-1])
        similarity_scores = torch.matmul(qs, ks.transpose(-2, -1))
        scaled_similarity_scores = similarity_scores / np.sqrt(d_k)
        squeezed_scores = self.softmax(scaled_similarity_scores)
        squeezed_scores = self.dropout(squeezed_scores)
        context_aware_scores = torch.matmul(squeezed_scores, vs)
        return context_aware_scores, squeezed_scores

    def forward(self, q, k, v):

        _Q = self.wq(q)
        _K = self.wk(k)
        _V = self.wv(v)

        qs = self.__create_heads(_Q)
        ks = self.__create_heads(_K)
        vs = self.__create_heads(_V)

        context_aware_scores, weights = self.__scaled_dot_product_attention(qs, ks, vs)
        output = self.__concatenate_heads(context_aware_scores)
        # TODO make sure of retyred values for visualisation purposes
        return self.output(output), context_aware_scores


class MLP(nn.Module):
    def __init__(self, config):
        super(MLP, self).__init__()

        # TODO __implement class properties

    @classmethod
    def from_config(cls, config):
        return cls(config)

    def forward(self, x):
        # TODO __implement forward pass
        pass
