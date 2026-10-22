"""Pasma v3 architecture; raw outputs are type/coarse/material logits."""
from typing import Any
import torch
import torch.nn as nn

TYPES = ["T_SHIRT", "SHIRT", "BLOUSE", "SWEATER", "HOODIE", "PANTS", "JEANS", "SHORTS", "SKIRT", "JACKET", "COAT", "VEST", "DRESS", "SHOES", "BOOTS", "SNEAKERS", "SANDALS", "ACCESSORY", "HAT", "SCARF"]
COARSE = ["TOP", "BOTTOM", "OUTERWEAR", "FOOTWEAR", "FULL_BODY", "ACCESSORY"]
MATERIALS = ["cotton", "linen", "wool", "cashmere", "silk", "denim", "corduroy", "tweed", "polyester", "nylon", "spandex_elastane", "acrylic", "rayon_viscose", "fleece", "technical_synthetic", "leather_smooth", "leather_suede", "leather_patent", "rubber", "metal_hardware", "blend_unspecified"]

def build_model(backbone: str, pretrained: bool = False, drop: float = 0.2, drop_path: float = 0.1):
    import timm

    class WardrobeNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            kwargs: dict[str, Any] = {"pretrained": pretrained, "num_classes": 0}
            if drop_path > 0:
                kwargs["drop_path_rate"] = drop_path
            self.backbone = timm.create_model(backbone, **kwargs)
            nf = getattr(self.backbone, "head_hidden_size", None) or self.backbone.num_features
            self.drop = nn.Dropout(drop)
            self.type_head = nn.Linear(nf, len(TYPES))
            self.coarse_head = nn.Linear(nf, len(COARSE))
            self.material_head = nn.Linear(nf, len(MATERIALS))

        def forward(self, x: torch.Tensor, return_feats: bool = False):
            feats = self.backbone(x)
            f = self.drop(feats)
            out = (self.type_head(f), self.coarse_head(f), self.material_head(f))
            return (*out, feats) if return_feats else out

    return WardrobeNet()
