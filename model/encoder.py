import torch.nn as nn
from model import register
from utils import MultiHeadSelfAttention


class ViTEncoder(nn.Module):
    def __init__(self, config, *, emb_dim: int):
        super(ViTEncoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock.from_config(config, emb_dim=emb_dim) for _ in range(config['ENCODER_LAYERS'])])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        pass


class EncoderBlock(nn.Module):
    def __init__(self, config, *, emb_dim: int):
        super(EncoderBlock, self).__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention.from_config(config, emb_dim=emb_dim)

        # TODO __add FFN
        self.ffn = None
        # TODO __add res conn.
        self.add_norm = None

    @classmethod
    def from_config(cls, config, *, emb_dim: int):
        return cls(config, emb_dim=emb_dim)

@register
def get_encoder(config, **kwargs):
    return ViTEncoder(config, emb_dim=kwargs['emb_dim'])
