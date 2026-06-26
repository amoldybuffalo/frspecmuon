from tinygpt import TinyGPT2, TinyGPT2Config, Tokenizer, GPTLanguageModel
from riemann_layers import riemannize
from r_optimizer import FrSpecMuon, SpecMuon
from optimizer_benchmark import benchmark_optimizers
import re
import copy
import torch
import matplotlib.pyplot as plt

tokenizer = Tokenizer()



#I'm just using the KJV bible as a sample finetuning text. I couldn't find any canonical corpus to do this on so I thought the bible would be fine.  
with open("kjv.txt", "r") as f:

    bible = f.read()

    # remove verse numbers and punctuation
    bible = re.sub(r"^.*?\d+:\d+\s*", "", bible, flags=re.MULTILINE)
    bible = re.sub(r"\[", "", bible)
    bible = re.sub(r"\]", "", bible)
    bible = re.sub(r"\,", "", bible)
    bible = re.sub(r"\;", "", bible)
    adamw_final_losses = []
    frspecmuon_final_losses = []
    specmuon_final_losses = []

    tokens = torch.tensor(
        tokenizer.encode(bible[:100_000]),
        dtype=torch.long,
    ).to(device)

 
    for i in range(10):
        device = "cuda:0"

        # Finetuning TinyGPT2 seems like a reasonable benchmark task
        model_frspecmuon = TinyGPT2.from_pretrained(
            "tinygpt2_ckpt_2026_02_18_20_42.pth"
        ).to(device)

        # Turns the linear layers of the model into LoRAed versions
        riemannize(model_frspecmuon, 60, exclusions=[model_frspecmuon.lm_head])

        model_specmuon = copy.deepcopy(model_frspecmuon)
        model_adamw = copy.deepcopy(model_frspecmuon)


        frspecmuon = FrSpecMuon(
            model_frspecmuon,
            lr=0.01,
            betas = (0.9,0.95)
        )

        hidden_weights = [p for p in model_specmuon.blocks.parameters() if p.ndim >= 2]
        hidden_gains_biases = [p for p in model_specmuon.blocks.parameters() if p.ndim < 2]
        nonhidden_params = [param for param in model_specmuon.lm_head.parameters()]
        param_groups = [
            dict(params=hidden_weights, use_muon=True),
            dict(params=hidden_gains_biases+nonhidden_params, use_muon=False),
        ]

        specmuon = SpecMuon(param_groups,
            lr = 0.01,
            betas = (0.9,0.95)
        )


        adamw = torch.optim.AdamW(
            model_adamw.parameters(),
            lr=3e-4,
            betas = (0.9, 0.95)
        )    


        final_losses = benchmark_optimizers([
            {"optimizer": frspecmuon, "label": "FrSpecMuon", "model": model_frspecmuon, "uses_closure": True}, 
            {"optimizer": specmuon, "label": "SpecMuon", "model": model_specmuon, "uses_closure": True},
            {"optimizer": adamw, "label": "AdamW", "model": model_adamw, "uses_closure": True}
            ],
            tokens,
            epoch_count = 20,
            steps_per_epoch = 100,
            graph = True,
            tag = str(i)
        )
        frspecmuon_final_losses.append(final_losses[0].item())
        specmuon_final_losses.append(final_losses[1].item())
        adamw_final_losses.append(final_losses[2].item())

    print(f"Final losses FrSpecmuon: {frspecmuon_final_losses}")
    print(f"Final losses Specmuon: {specmuon_final_losses}")
    print(f"Final losses AdamW: {adamw_final_losses}")



        












