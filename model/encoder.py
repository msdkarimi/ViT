import torch.nn as nn
from model import register
from utils import MultiHeadSelfAttention


class ViTEncoder(nn.Module):
    def __init__(self, config, *, emb_dim: int):
        super(ViTEncoder, self).__init__()

        self.layers = nn.ModuleList([
            EncoderLayer.from_config(config, emb_dim=emb_dim)
            for _ in range(config['ENCODER_LAYERS'])
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)


class EncoderLayer(nn.Module):
    def __init__(self, config, *, emb_dim: int):
        super(EncoderLayer, self).__init__()
        self.multi_head_self_attention = MultiHeadSelfAttention.from_config(config, emb_dim=emb_dim)
        self.mlp = MLP.from_config(config, emb_dim=emb_dim)
        self.norm1 = nn.LayerNorm(emb_dim)
        self.norm2 = nn.LayerNorm(emb_dim)

    @classmethod
    def from_config(cls, config, *, emb_dim: int):
        return cls(config, emb_dim=emb_dim)

    def forward(self, x):
        x = x + self.multi_head_self_attention(*[self.norm1(x)]*3)
        return x + self.mlp(self.norm2(x))


class MLP(nn.Module):
    def __init__(self, *, emb_dim: int, ff_hidden_layer_scale: int, dropout: float = None):
        super(MLP, self).__init__()

        self.fcl1 = nn.Linear(emb_dim, ff_hidden_layer_scale * emb_dim)
        self.fcl2 = nn.Linear(ff_hidden_layer_scale * emb_dim, emb_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    @classmethod
    def from_config(cls, config, *, emb_dim: int):
        return cls(emb_dim=emb_dim, ff_hidden_layer_scale=config['HIDDEN_LAYER_SCALE'], dropout=config['DROP_OUT'])

    def forward(self, x):
        return self.fcl2(self.dropout(self.relu(self.fcl1(x))))


@register
def get_encoder(config, **kwargs):
    return ViTEncoder(config, emb_dim=kwargs['emb_dim'])
