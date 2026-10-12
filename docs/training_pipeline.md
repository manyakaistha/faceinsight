# Training Pipeline

Faithful reproduction of original notebooks:
- LoRAConv2d on 3x3 convs (ResNet50 Bottlenecks)
- Loss: 0.7*Huber(delta=5.0) + 0.3*BCEWithLogits(label_smooth=0.05)
- EMA decay 0.9995, TTA (flip), RandAugment + ColorJitter + RandomResizedCrop
- Externalized configs in configs/*.yaml
