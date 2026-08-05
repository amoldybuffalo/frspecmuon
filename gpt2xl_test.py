from transformers import AutoTokenizer, AutoModelForCausalLM
from frspecmuon import FrSpecMuon
from optimizer_benchmark import benchmark_optimizers
from datasets import RandomLanguageModelDataset
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
from frspecmuon import FrSpecMuon, FrSpecMuon_USVh, FrSpecMuon_with_momentum
from riemann_layers import riemannize, split_parameters
from optimizer_benchmark import benchmark_optimizers_resnet, benchmark_optimizers
from muon import SingleDeviceMuonWithAuxAdam
from torch.optim import AdamW

DEVICE = "cuda:0"

model_name = "openai-community/gpt2-medium"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


# #I'm just using the KJV bible as a sample finetuning text. I couldn't find any canonical corpus to do this on so I thought the bible would be fine.  
# with open("kjv.txt", "r") as f:

#     bible = f.read()

#     # remove verse numbers and punctuation
#     bible = re.sub(r"^.*?\d+:\d+\s*", "", bible, flags=re.MULTILINE)
#     bible = re.sub(r"\[", "", bible)
#     bible = re.sub(r"\]", "", bible)
#     bible = re.sub(r"\,", "", bible)
#     bible = re.sub(r"\;", "", bible)

# with open("romeo_and_juliet.txt", "r") as f:
#     romeo_and_juliet = f.read()

# tokens = tokenizer(romeo_and_juliet[:len(romeo_and_juliet)//2],  return_tensors="pt")


device = "cuda:0"

def get_gpt_trainloader(model_name, text_file, steps_per_epoch = 100):

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    with open(text_file, "r") as f:
        text = f.read()

    tokens = tokenizer(text,  return_tensors="pt")["input_ids"][0]    

    trainloader = DataLoader(
    RandomLanguageModelDataset(tokens, 1024, steps_per_epoch),
    batch_size=1,
    )

    return trainloader

def get_gpt_model(model_name):
    model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
    )

    # Turns the linear layers of the model into LoRAed versions
    riemannize(model, 20, exclusions=[model.lm_head], mode="USVh").to("cpu")
    return model

def init_frspecmuon(model, hyperparameters):
        return FrSpecMuon(model, **hyperparameters)

def init_frspecmuon_usvh(model, hyperparameters):
        return FrSpecMuon_USVh(model, **hyperparameters)


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




def loss_fn_gpt(model, batch):
    inputs = batch.to(DEVICE)
    return model(
        input_ids=inputs,
        labels=inputs,
    ).loss



if __name__ == "__main__":
    
 


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
                "lr": 0.1,
                "weight_decay":0.00
            },

            "label": "FrSpecMuon",
            "optimizer_fn": init_frspecmuon_momentum,
            "uses_closure": True,
        },
        {
            "hyperparameters": {
                "lr": 0.0003,
                "betas": (0.9, 0.999),
                "weight_decay": 0.0
          
            },
            "label": "AdamW",
            "optimizer_fn": init_adamw,
            "uses_closure": True,
        },

        {
            "hyperparameters": {
                "lr": 0.001,
                "momentum":0.95,
                "weight_decay":0.01
            },
            "label": "Muon",
            "optimizer_fn": init_muon,
            "uses_closure": True,
        },
    ]

    #########################################################
    # Benchmark
    #########################################################

    trainloader = get_gpt_trainloader(model_name, "romeo_and_juliet.txt", 250)
    model = get_gpt_model(model_name)

    benchmark_optimizers(
        model,
        optimizer_configs,
        trainloader,
        loss_fn_gpt,
        device=device,
        epoch_count=30,
        graph=True,
        graph_type = "epochs",
        graph_output_dir="graphs/gpt",
        tag="GPT_TEST",
    )   




