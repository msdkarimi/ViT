from model import register
import torch.nn as nn
import torch
from model import build
from utils.model_utils import PatchEmbedding, MLP


class VisionTransformer(nn.Module):
    def __init__(self, config: dict):
        super(VisionTransformer, self).__init__()

        self.extracted_features_emb = nn.Sequential(
            PatchEmbedding.from_config(config),
            build(config['ENCODER'], config['ENCODER']['NAME'], emb_dim=config['INPUT_CHANNEL'] * config['PATCH_DIM'] ** 2)
        )
        self.classifier = nn.Sequential(MLP.from_config(config['CLASSIFIER']), nn.Sigmoid())

    def forward(self, x):
        x = self.extracted_features_emb(x)
        # TODO __make sure of dimension of x
        return self.classifier(x[0, :])


@register
def get_vit(config: dict):
    return VisionTransformer(config['MODEL'])
