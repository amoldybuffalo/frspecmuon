from tinygpt import TinyGPT2, TinyGPT2Config, Tokenizer, GPTLanguageModel
from matplotlib import pyplot as plt
import torch
from riemann_layers import riemannize
from r_optimizer import FrSpecMuon
import re

tokenizer = Tokenizer()

device = "cuda:0"

model = TinyGPT2.from_pretrained(
    "tinygpt2_ckpt_2026_02_18_20_42.pth"
).to(device)

# Turns the linear layers of the model into LoRAed versions
riemannize(model, 60, exclusions=[model.lm_head])

frspecmuon = FrSpecMuon(
    model,
    lr=0.001,
    betas = (0,0),
    debug = True
)


step_count = 10000

#I'm just using the KJV bible as a sample finetuning text. I couldn't find any canonical corpus to do this on so I thought the bible would be fine.  
with open("kjv.txt", "r") as f:

    bible = f.read()

    # remove verse numbers and punctuation
    bible = re.sub(r"^.*?\d+:\d+\s*", "", bible, flags=re.MULTILINE)
    bible = re.sub(r"\[", "", bible)
    bible = re.sub(r"\]", "", bible)
    bible = re.sub(r"\,", "", bible)
    bible = re.sub(r"\;", "", bible)

    tokens = torch.tensor(
        tokenizer.encode(bible),
        dtype=torch.long,
    ).to(device)

    r_squares = []
    losses = []

    for step in range(step_count):
        token_index = torch.randint(
            512,
            len(tokens) - 512,
            (),
            device=device,
        ).item()

        def closure():
            logits, loss, _ = model(
                            tokens[token_index - block_size:token_index][None, :],
                            tokens[token_index:token_index + block_size][None, :],
                        )

            loss.backward()

            return loss

        loss, r = frspecmuon.step(closure)
        r_2 = r**2
        r_squares.append(r_2.item())
        losses.append(loss.item())

        print(f"progress: {100*step/step_count:0.2f}% loss: {loss:0.3f} r^2: {r_2}   ", end='\r')

    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, step_count+1),
        losses,
        linewidth=2,
        label="loss",
    )

    plt.plot(
        range(1, step_count+1),
        r_squares,
        linewidth=2,
        label="r^2",
    )


    plt.xlabel("Step")

    plot_title = "R^2 and Loss plot"

    plt.title(plot_title)
    plt.grid(True)
    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{plot_title}.png",
        dpi=300,
    )

    plt.show()    