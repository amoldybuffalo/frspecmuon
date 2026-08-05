from simple_param_search import grid_search
from datasets import RandomLanguageModelDataset
import copy
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import datasets, transforms
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader
from frspecmuon import FrSpecMuon, FrSpecMuon_USVh
from riemann_layers import riemannize, riemannize, split_parameters
from optimizer_benchmark import benchmark_optimizers_resnet, benchmark_optimizers
from muon import SingleDeviceMuonWithAuxAdam
from torch.optim import AdamW
import numpy as np
import json

DEVICE = "cuda:0"

def get_gpt_trainloader(model_name, text_file, steps_per_epoch = 1000):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    with open(text_file, "r") as f:
        text = f.read()

    tokens = tokenizer(text,  return_tensors="pt")["input_ids"][0]    

    trainloader = DataLoader(
    RandomLanguageModelDataset(tokens, 1024, steps_per_epoch),
    batch_size=2,
    )

    return trainloader

def get_resnet_trainloader():
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Optional if using ImageNet ResNet
    transforms.ToTensor(),
    transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )    # Turns the linear layers of the model into LoRAed versions
    ])

    train_dataset = datasets.CIFAR100(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )


    trainloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    return trainloader


def get_resnet_model():
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    num_classes = 100
    model = models.resnet50(weights=weights).to("cpu")
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    riemannize(model, 20, exclusions=[model.lm_head], mode="USVh")
    return model

def get_gpt_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    ).to("cpu")

    # Turns the linear layers of the model into LoRAed versions
    riemannize(model, 20, exclusions=[model.lm_head], mode="USVh")

    return model

def init_frspecmuon(model, hyperparameters):
        return FrSpecMuon(model, **hyperparameters)

def init_frspecmuon_usvh(model, hyperparameters):
    return FrSpecMuon_USVh(model, **hyperparameters)

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




def loss_fn_gpt(model, batch):
    inputs = batch.to(DEVICE)
    return model(
        input_ids=inputs,
        labels=inputs,
    ).loss

cross_entropy = nn.CrossEntropyLoss()

def loss_fn_resnet(model, batch):
    images, labels = batch
    images = images.to(DEVICE)
    labels = labels.to(DEVICE)
    return cross_entropy(model(images), labels)



from itertools import product
import copy

SEARCH_SPACES = {
    "ResNet": {
        "AdamW": {
            "optimizer_fn": init_adamw,
            "grid": {
                "lr": [1e-4, 3e-4, 1e-3],
                "weight_decay": [0.0, 0.01],
            },
        },

        "Muon": {
            "optimizer_fn": init_muon,
            "grid": {
                "lr": [0.001, 0.002, 0.0005],
                "momentum": [0.90, 0.95],
                "weight_decay": [0.0, 0.01],
            },
        },

        "FrSpecMuon": {
            "optimizer_fn": init_frspecmuon_usvh,
            "grid": {
                "lr": [0.001, 0.005, 0.01],
                "weight_decay": [0.0, 0.01],
            },
        },
    },

    "GPT": {
        "AdamW": {
            "optimizer_fn": init_adamw,
            "grid": {
                "lr": [3e-5, 1e-4, 3e-4],
                "weight_decay": [0.0, 0.01],
            },
        },

        "Muon": {
            "optimizer_fn": init_muon,
            "grid": {
                "lr": [0.005, 0.001, 0.002],
                "momentum": [0.90, 0.95],
                "weight_decay": [0.0, 0.01],
            },
        },

        "FrSpecMuon": {
            "optimizer_fn": init_frspecmuon_usvh,
            "grid": {
                "lr": [0.1, 0.3, 0.05],
                "weight_decay": [0.0, 0.01],
            },
        },
    },
}

def test_resnet():
    print("TESTING RESNET MODEL")
    resnet_loader = get_resnet_trainloader()
    resnet = get_resnet_model()

    best_params = {}
    print("Finding the best parameters for each optimizer:")

    for optimizer_name, config in SEARCH_SPACES["ResNet"].items():
        print(f"Finding for {optimizer_name}:")
        result = grid_search(
            base_model=resnet,
            trainloader=resnet_loader,
            loss_fn=loss_fn_resnet,
            optimizer_fn=config["optimizer_fn"],
            grid=config["grid"],
            epoch_count=4,
            benchmark_fn=benchmark_optimizers,
            grid_batch_size=2
        )

        best_params[optimizer_name] = result["best_hyperparameters"]
        with open("best_params_resnet.json", "w") as f:
            json.dump(best_params, f)        

        print(f"Best parameters for {optimizer_name } are: ")
        print(best_params[optimizer_name])


    print("Doing a final test: \n")


    optimizer_configs = [
        {
            "hyperparameters": best_params["FrSpecMuon"],

            "label": "FrSpecMuon",
            "optimizer_fn": init_frspecmuon,
            "uses_closure": True,
        },
        {
            "hyperparameters": best_params["AdamW"],
            "label": "AdamW",
            "optimizer_fn": init_adamw,
            "uses_closure": True,
        },

        {
            "hyperparameters": best_params["Muon"],
            "label": "Muon",
            "optimizer_fn": init_muon,
            "uses_closure": True,
        },
    ]

    benchmark_optimizers(
        resnet,
        optimizer_configs,
        resnet_loader,
        loss_fn_resnet,
        device=DEVICE,
        epoch_count=10,
        graph=True,
        graph_type = "epochs",
        graph_output_dir="graphs/resnet",
        tag="FINAL_TEST_RESNET",
    )     



def test_gpt():
    print("TESTING GPT MODEL")
    gpt_loader = get_gpt_trainloader("openai-community/gpt2-medium", "romeo_and_juliet.txt")
    gpt = get_gpt_model("openai-community/gpt2-medium")

    best_params = {}
    print("Finding the best parameters for each optimizer:")

    for optimizer_name, config in SEARCH_SPACES["GPT"].items():
        print(f"Finding for {optimizer_name}:")
        result = grid_search(
            base_model=gpt,
            trainloader=gpt_loader,
            loss_fn=loss_fn_gpt,
            optimizer_fn=config["optimizer_fn"],
            grid=config["grid"],
            epoch_count=4,
            benchmark_fn=benchmark_optimizers,
            grid_batch_size=2
        )

        best_params[optimizer_name] = result["best_hyperparameters"]

        print(f"Best parameters for {optimizer_name } are: ")
        print(best_params[optimizer_name])
        with open("best_params_gpt.json", "w") as f:
            json.dump(best_params, f)

    
    
   

    print("Doing a final test: \n")


    optimizer_configs = [
        {
            "hyperparameters": best_params["FrSpecMuon"],

            "label": "FrSpecMuon",
            "optimizer_fn": init_frspecmuon,
            "uses_closure": True,
        },
        {
            "hyperparameters": best_params["AdamW"],
            "label": "AdamW",
            "optimizer_fn": init_adamw,
            "uses_closure": True,
        },

        {
            "hyperparameters": best_params["Muon"],
            "label": "Muon",
            "optimizer_fn": init_muon,
            "uses_closure": True,
        },
    ]

    benchmark_optimizers(
        gpt,
        optimizer_configs,
        gpt_loader,
        loss_fn_gpt,
        device=DEVICE,
        epoch_count=10,
        graph=True,
        graph_type = "epochs",
        graph_output_dir="graphs/gpt",
        tag="FINAL_TEST_GPT",
    )

if __name__ == "__main__":
    test_gpt()
    test_resnet()