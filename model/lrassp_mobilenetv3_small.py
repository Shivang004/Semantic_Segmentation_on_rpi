import torch
from collections import OrderedDict
from functools import partial
from typing import Any, Dict, Optional
from torch import nn, Tensor
from torch.nn import functional as F
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.mobilenetv3 import mobilenet_v3_small

__all__ = ["LRASPP", "LRASPPHead", "load_lraspp_mobilenet_v3_small"]

class LRASPP(nn.Module):
    def __init__(self, backbone: nn.Module, low_channels: int, high_channels: int, num_classes: int, inter_channels: int = 128) -> None:
        super().__init__()
        self.backbone = backbone
        self.classifier = LRASPPHead(low_channels, high_channels, num_classes, inter_channels)

    def forward(self, input: Tensor) -> Dict[str, Tensor]:
        features = self.backbone(input)
        out = self.classifier(features)
        out = F.interpolate(out, size=input.shape[-2:], mode="bilinear", align_corners=False)

        result = OrderedDict()
        result["out"] = out

        return result

class LRASPPHead(nn.Module):
    def __init__(self, low_channels: int, high_channels: int, num_classes: int, inter_channels: int) -> None:
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),
        )
        self.scale = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(high_channels, inter_channels, 1, bias=False),
            nn.Sigmoid(),
        )
        self.low_classifier = nn.Conv2d(low_channels, num_classes, 1)
        self.high_classifier = nn.Conv2d(inter_channels, num_classes, 1)

    def forward(self, input: Dict[str, Tensor]) -> Tensor:
        low = input["low"]
        high = input["high"]

        x = self.cbr(high)
        s = self.scale(high)
        x = x * s
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)

        return self.low_classifier(low) + self.high_classifier(x)

def _lraspp_mobilenetv3(backbone: nn.Module, num_classes: int) -> LRASPP:
    backbone = backbone.features
    stage_indices = [0] + [i for i, b in enumerate(backbone) if getattr(b, "_is_cn", False)] + [len(backbone) - 1]
    low_pos = stage_indices[-4]
    high_pos = stage_indices[-1]
    low_channels = backbone[low_pos].out_channels
    high_channels = backbone[high_pos].out_channels
    backbone = IntermediateLayerGetter(backbone, return_layers={str(low_pos): "low", str(high_pos): "high"})

    return LRASPP(backbone, low_channels, high_channels, num_classes)

def load_lraspp_mobilenet_v3_small(checkpoint_path: str, num_classes: int, device: Optional[torch.device] = None) -> LRASPP:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    backbone = mobilenet_v3_small(dilated=True)
    model = _lraspp_mobilenetv3(backbone, num_classes)
    pretrained_dict=torch.load(checkpoint_path, map_location=device)['model']
    model_dict=model.state_dict()
    # Filter out unnecessary keys (excluding final classifier layer)
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'classifier' not in k}

    # Load the pretrained weights into the model
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model

