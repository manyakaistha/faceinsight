# Repository Audit

This source repository omits notebooks, private face data, and model weight files. Supply the champion artifacts locally as described in the README to enable inference.

## Identified Champion Architecture
- Checkpoint: `export_resnet18_lora_v6/` (best: `checkpoint_e22_s0.9212.pth`, score 0.9212)
- Backbone: ResNet50 (as confirmed by checkpoint key names and `model_meta.json.arch = resnet50`)
- Feature dim: 2048 (layer4 outputs)
- LoRA: Conv2d adapters with `r=12`, `alpha=24` injected into Bottleneck conv1/conv2/conv3 and downsample conv layers (including first conv). Scaling = alpha/r.
- Heads: Age (Dropout(0.2)->Linear(2048,256)->ReLU->Linear(256,1)), Gender (same) with logits output.
- Training: Huber loss (delta 5.0) for age, label-smoothed BCE (eps 0.05) for gender, weighted loss age_weight=0.7, gender_weight=0.3; EMA decay 0.9995; TTA (horizontal flip); normalization ImageNet [0.485,0.456,0.406]/[0.229,0.224,0.225].

## Checkpoint Compatibility
- Keys match ResNet50 Bottleneck naming (layerX.N.conv1/conv2/conv3, downsample). `out_dim` 2048 matches ResNet50.
- Weights load successfully with strict=False acceptable.

## Preprocessing
- Resize to 224x224, ImageNet normalization; EXIF orientation handled.

## Discrepancies
- Directory name `export_resnet18_lora_v6` is misleading (architecture is ResNet50). Meta and weights confirm ResNet50.

## Technical Debt
- Notebooks contain large experiment history; production code modularized. Face detection optional (MediaPipe/HAAR fallback).

## Application Architecture
- `ml/` contains models, inference, evaluation, explainability.
- `backend/` FastAPI exposing `/api/v1/*`.
- Next.js research studio with live prediction, Grad-CAM explanations, evaluation reports, and architecture explorer.

## Remaining Artifacts
- Evaluation metrics on small validation splits produced (JSON). Larger runs can be performed offline.
