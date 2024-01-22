from model import ViT
from model_builder import Registry


model = Registry.from_conf_create_model('ViT', {'heads': 8}, name='david')
print(model.heads)
