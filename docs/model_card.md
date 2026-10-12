# Model Card - FaceInsight AI (NPPE-1)

## Intended Use
Multi-task facial age estimation (0-100) and binary gender classification (0=Female,1=Male as per dataset labels) from 224x224 RGB face crops.

## Model Details
- Architecture: ResNet50 + LoRAConv2d (r=12, α=24), two MLP heads (2048->256->1). < 15% trainable params.
- Checkpoint: export_resnet18_lora_v6/checkpoint_e22_s0.9212.pth (champion). Directory name is legacy but arch confirmed ResNet50.

## Training Data
~34,708 labeled facial images; class imbalance ~75.2% Male, ~24.8% Female. Age range ~1-100.

## Metrics
- Gender: Macro-F1; Age: RMSE; Composite = 2*F1*age_quality/(F1+age_quality), age_quality = max(1-RMSE/100,1e-6). Champion report ~0.9212 (competition proxy). Validation samples on small splits included in artifacts/.

## Limitations & Responsible Use
- Labels reflect dataset taxonomy (0=Female,1=Male); output is an estimate based on visual patterns learned from training data - not a determination of gender identity/biological sex.
- Performance varies with image quality, pose, occlusion, illumination.
- No fairness claims without suitable demographic evaluation.
- Not for high-stakes decisions.
