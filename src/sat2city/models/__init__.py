"""Model factory: `model:` in a training config is a Hugging Face DAv2 id or `unet[:encoder]`."""

from __future__ import annotations


def build_model(name: str, cache_dir: str | None = None, pretrained: bool = True):
    if name.startswith("unet"):
        from sat2city.models.unet import UNetHeight

        encoder = name.split(":", 1)[1] if ":" in name else "resnet34"
        return UNetHeight(encoder=encoder, pretrained=pretrained)
    from sat2city.models.dav2 import DAv2Height

    return DAv2Height(name, cache_dir=cache_dir)
