from model import BaseModel


class ViT(BaseModel, ):
    def __init__(self, config: dict, *, name=None):
        super(ViT, self).__init__(config)
        self.name = name
