from model import BaseModel
from utils.model_utils import PatchEmbedding


class ViT(BaseModel, ):
    def __init__(self, config: dict):
        super(ViT, self).__init__(config)

        self.patch_embedding = PatchEmbedding(img_dim=(self.cfg['INPUT_H'], self.cfg['INPUT_W']), input_channel=self.cfg['INPUT_CHANNEL'], patch_dim=self.cfg['PATCH_DIM'])


    def forward(self, x):
        x = self.patch_embedding(x)