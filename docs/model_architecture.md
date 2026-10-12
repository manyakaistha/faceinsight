# Model Architecture (Recovered, Faithful)

- Backbone: ResNet50 (torchvision) pretrained on ImageNet, final fc replaced by Identity.
- LoRAConv2d: wraps frozen Conv2d with A,B 1x1; scaling α/r. Injected into conv1, all Bottleneck conv1/conv2/conv3, and downsample conv (where present).
- Shared features: 2048-d.
- Age head: Linear(2048,256)-ReLU-Linear(256,1); regression clamped to [0,100].
- Gender head: Linear(2048,256)-ReLU-Linear(256,1); logits -> sigmoid; threshold (default 0.5; can be tuned).
- Loss: L_total = 0.7*Huber(age, δ=5.0) + 0.3*BCEWithLogitsWithLabelSmoothing(gender, ε=0.05).
- Inference: TTA optional (average original+horizontal flip).
