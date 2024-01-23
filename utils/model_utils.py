import torch
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

    def forward(self, x):
        x = self.perform_patching_on_input(x).flatten(2).transpose(1, 2)  # b_s, num_patches, emb_dim
        b_s, _, _ = x.shape
        cls_tokes = self.cls_token.expand(b_s, -1, -1)
        x = torch.cat((cls_tokes, x), dim=1)
        x += self.positional_encoding
        return x
