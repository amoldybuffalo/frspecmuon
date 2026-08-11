import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageNet
from torchvision.transforms import v2
from imagenetv2_pytorch import ImageNetV2Dataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from r_optimizer import FrSpecMuon
from frspecmuon import FrSpecMuon, FrSpecMuon_with_momentum
from riemann_layers import riemannize, split_parameters
from optimizer_benchmark import benchmark_optimizers
from muon import SingleDeviceMuonWithAuxAdam
from torch.optim import AdamW
if __name__ == "__main__":
    device = "cuda:0"

    #########################################################
    # Pretrained ResNet
    #########################################################

    weights = models.ResNet50_Weights.IMAGENET1K_V1




    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # Optional if using ImageNet ResNet
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    train_dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    #########################################################
    # Create identical models
    #########################################################
    num_classes = 100
    model = models.resnet50(weights=weights).to(device)
    model.fc = nn.Linear(model.fc.in_features, num_classes).to(device)


    riemannize(model, 20, exclusions = [model.fc], mode="USVh")
    
    def init_frspecmuon(model, hyperparameters):
        return FrSpecMuon(model, **hyperparameters)
    
    def init_frspecmuon_momentum(model, hyperparameters):
        return FrSpecMuon_with_momentum(model, **hyperparameters)


    def init_muon(model, hyperparameters):
        riemann_params, other_params = split_parameters(model)

        param_groups = [
            dict(params=riemann_params, use_muon=True,
                 lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"], momentum=hyperparameters["momentum"]),
            dict(params=other_params, use_muon=False,
                 lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01),
        ]

        return SingleDeviceMuonWithAuxAdam(param_groups)


    def init_adamw(model, hyperparameters):
        return torch.optim.AdamW(model.parameters(), **hyperparameters)


    criterion = nn.CrossEntropyLoss()

    def loss_fn(model, batch):
        images, labels = batch
        images = images.to(device)
        labels = labels.to(device)
        return criterion(model(images), labels)
 


    #########################################################
    # Loss
    #########################################################

    criterion = nn.CrossEntropyLoss()

    #########################################################
    # Optimizers
    #########################################################

    optimizer_configs = [
        {
            "hyperparameters": {
                "lr": 0.01,
                "q_multiplier": 2,
                "relaxation_tolerance": 0.95,
                "weight_decay":0.00,
            },

            "label": "FrSpecMuon",
            "optimizer_fn": init_frspecmuon,
            "uses_closure": True,
        },

        {
            "hyperparameters": {
                "lr": 0.01,
                "q_multiplier": 2,
                "relaxation_tolerance": 0.95,
                "weight_decay":0.00,
                "betas": (0.9, 0.99)
            },

            "label": "FrSpecMuon (with momentum)",
            "optimizer_fn": init_frspecmuon_momentum,
            "uses_closure": True,
        },

        # {
        #     "hyperparameters": {
        #         "lr": 0.0003,
        #         "betas": (0.9, 0.999),
        #         "weight_decay":0.00
          
        #     },
        #     "label": "AdamW",
        #     "optimizer_fn": init_adamw,
        #     "uses_closure": True,
        # },

        # {
        #     "hyperparameters": {
        #         "lr": 0.001,
        #         "momentum":0.9,
        #         "weight_decay":0.00
        #     },
        #     "label": "Muon",
        #     "optimizer_fn": init_muon,
        #     "uses_closure": True,
        # },
    ]

    #########################################################
    # Benchmark
    #########################################################

    benchmark_optimizers(
        model,
        optimizer_configs,
        trainloader,
        loss_fn,
        device=device,
        epoch_count=30,
        graph=True,
        graph_type = "epochs",
        graph_output_dir="graphs/resnet",
        tag="second_momentum_only_decreased_lr",
    )     