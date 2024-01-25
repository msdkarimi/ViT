from model import register
import torch.nn as nn
from model import build
from utils.model_utils import PatchEmbedding


class VisionTransformer(nn.Module):
    def __init__(self, config: dict):
        super(VisionTransformer, self).__init__()

        self.projection = PatchEmbedding.from_config(config)
        self.encoder = build(config['ENCODER'], config['ENCODER']['NAME'], emb_dim=config['INPUT_CHANNEL']*config['PATCH_DIM'] ** 2)
        # TODO __implement MLP as classifier which takes first tokes self.feature_emb[0, :]
        self.classifier = None
        # self.feature_emb = nn.Sequential(self.projection, self.encoder)

    def forward(self, x):
        x = self.patch_embedding(x)


@register
def get_vit(config: dict):
    return VisionTransformer(config['MODEL'])
