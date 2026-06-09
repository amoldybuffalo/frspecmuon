from tinygpt import TinyGPT2, TinyGPT2Config, Tokenizer, GPTLanguageModel
from riemann_layers import riemannize
from r_optimizer import FrSpecMuon

import re
import copy
import torch
import matplotlib.pyplot as plt

tokenizer = Tokenizer()

device = "cuda:0"

model = TinyGPT2.from_pretrained(
    "tinygpt2_ckpt_2026_02_18_20_42.pth"
).to(device)

non_riemann_model = copy.deepcopy(model)

riemannize(model, 20, exclusions=[model.lm_head])

riemann_optimizer = FrSpecMuon(
    model,
    lr=1e-3,
)

nonriemann_optimizer = torch.optim.AdamW(
    non_riemann_model.parameters(),
    lr=1e-3,
)

riemann_losses = []
non_riemann_losses = []

with open("kjv.txt", "r") as f:

    bible = f.read()

    # remove verse numbers and punctuation
    bible = re.sub(r"^.*?\d+:\d+\s*", "", bible, flags=re.MULTILINE)
    bible = re.sub(r"\[", "", bible)
    bible = re.sub(r"\]", "", bible)
    bible = re.sub(r"\,", "", bible)
    bible = re.sub(r"\;", "", bible)

    bible_tokens = torch.tensor(
        tokenizer.encode(bible),
        dtype=torch.long,
    ).to(device)

    # Match approximately the same number of updates as before
    steps_per_epoch = 4000

    model.train()
    non_riemann_model.train()

    for epoch in range(15):

        avg_loss_riemann = 0.0
        avg_loss_nonriemann = 0.0
        count = 0

        print(f"\nEpoch {epoch + 1}")

        for step in range(steps_per_epoch):

            # Sample a random window
            i = torch.randint(
                512,
                len(bible_tokens) - 512,
                (),
                device=device,
            ).item()

            count += 1

            def closure():

                logits, loss, _ = model(
                    bible_tokens[i - 512:i][None, :],
                    bible_tokens[i:i + 512][None, :],
                )

                loss.backward()

                return loss

            loss_riemann = riemann_optimizer.step(closure)

            avg_loss_riemann += loss_riemann.item()

            logits, loss_nonriemann, _ = non_riemann_model(
                bible_tokens[i - 512:i][None, :],
                bible_tokens[i:i + 512][None, :],
            )

            loss_nonriemann.backward()

            nonriemann_optimizer.step()

            avg_loss_nonriemann += loss_nonriemann.item()

            nonriemann_optimizer.zero_grad()
            riemann_optimizer.zero_grad()

            print(
                f"progress: {100 * count / steps_per_epoch:0.2f}% "
                f"current loss: {loss_riemann.item():0.4f} "
                f"avg riemann: {avg_loss_riemann / count:0.4f} "
                f"avg adamw: {avg_loss_nonriemann / count:0.4f}",
                end="\r",
            )

        avg_loss_riemann /= count
        avg_loss_nonriemann /= count

        riemann_losses.append(avg_loss_riemann)
        non_riemann_losses.append(avg_loss_nonriemann)

        print(
            f"\nEpoch {epoch + 1} complete:"
            f"\n  Riemann avg loss:     {avg_loss_riemann:.4f}"
            f"\n  AdamW avg loss:       {avg_loss_nonriemann:.4f}"
        )

        torch.save(model.state_dict(), f"checkpoints_riemann/{epoch}.pt")
        torch.save(non_riemann_model.state_dict(), f"checkpoints_adamw/{epoch}.pt")
my_input = torch.tensor(
    tokenizer.encode("The lord said  "),
    dtype=torch.long,
).to(device)

print("\n\nAfter training on bible data (Riemann):")
print(
    tokenizer.decode(
        model.generate(
            my_input[None, :],
            100,
            temperature=1.0,
        )[0].tolist()
    )
)

print("\n\nAfter training on bible data (AdamW):")
print(
    tokenizer.decode(
        non_riemann_model.generate(
            my_input[None, :],
            100,
            temperature=1.0,
        )[0].tolist()
    )
)

plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(riemann_losses) + 1),
    riemann_losses,
    linewidth=2,
    label="Riemann",
)

plt.plot(
    range(1, len(non_riemann_losses) + 1),
    non_riemann_losses,
    linewidth=2,
    label="AdamW",
)

plt.xlabel("Epoch")
plt.ylabel("Average Loss")
plt.title("Riemann vs AdamW Fine-Tuning")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    "riemann_vs_nonriemann_loss.png",
    dpi=300,
)

plt.show()









