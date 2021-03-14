import torch.nn as nn
from exceptions.exceptions import InvalidBackboneError
from models.resnet import ResNet18, ResNet50


class ResNetSimCLR(nn.Module):
    def __init__(self, base_model, out_dim):
        super(ResNetSimCLR, self).__init__()
        self.resnet_dict = {"resnet18": ResNet18(num_classes=out_dim),
                            "resnet50": ResNet50(num_classes=out_dim)}

        self.backbone = self._get_basemodel(base_model)

    def _get_basemodel(self, model_name):
        try:
            model = self.resnet_dict[model_name]
        except KeyError:
            raise InvalidBackboneError(
                "Invalid backbone architecture. Check the config file and pass one of: resnet18 or resnet50")
        else:
            return model

    def forward(self, x):
        return self.backbone(x)
