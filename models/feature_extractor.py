import timm
from torch import nn


def create_backbone(model_name: str):
    backbone = timm.create_model(model_name, pretrained=True)
    for param in backbone.parameters():
        backbone.requires_grad = False
    backbone.fc = nn.Identity()
    return backbone
