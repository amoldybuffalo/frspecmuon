import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageNet
from torchvision.transforms import v2
from imagenetv2_pytorch import ImageNetV2Dataset
from r_optimizer import FrSpecMuon
from riemann_layers import riemannize, riemannize_experimental
from optimizer_benchmark import benchmark_optimizers_resnet

device = "cuda:0"

#########################################################
# Pretrained ResNet
#########################################################

weights = models.ResNet50_Weights.IMAGENET1K_V1

transform = weights.transforms()

#########################################################
# ImageNet-V2 dataset
#########################################################

train_dataset = ImageNetV2Dataset("matched-frequency",
    transform=transform,
)

trainloader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=True,
)

#########################################################
# Create identical models
#########################################################

model_frspecmuon_momentum = models.resnet50(weights=weights).to(device)

riemannize_experimental(model_frspecmuon_momentum, 20, exclusions = [model_frspecmuon_momentum.fc])

# model_adamw = copy.deepcopy(model_frspecmuon)
model_frspecmuon_no_momentum = copy.deepcopy(model_frspecmuon_momentum)

frspecmuon_momentum = FrSpecMuon(
    model_frspecmuon_momentum,
    lr=3e-4,
    betas = (0.9,0.999),
    weight_decay = 0.00
)

frspecmuon_no_momentum = FrSpecMuon(
    model_frspecmuon_no_momentum,
    lr=3e-4,
    betas = (0,0),
    weight_decay = 0.00
)

# adamw = torch.optim.AdamW(
#     model_adamw.parameters(),
#     lr=3e-4,
#     betas = (0.0,0.0)
# )   

#########################################################
# Loss
#########################################################

criterion = nn.CrossEntropyLoss()

#########################################################
# Optimizers
#########################################################

optimizer_configs = [
    {
        "model": model_frspecmuon_momentum,
        "optimizer": frspecmuon_momentum,
        "label": "FrSpecMuon (momentum)",
        "uses_closure": True,
    },
    {
        "model": model_frspecmuon_no_momentum,
        "optimizer": frspecmuon_no_momentum,
        "label": "FrSpecmuon (no momentum)",
        "uses_closure": True,
    },
]

#########################################################
# Benchmark
#########################################################

benchmark_optimizers_resnet(
    optimizer_configs,
    trainloader,
    criterion,
    device=device,
    epoch_count=10,
    graph=True,
    tag="imagenetv2",
)