from transformers import AutoTokenizer, AutoModelForCausalLM
from riemann_layers import riemannize, riemannize_experimental
from r_optimizer import FrSpecMuon, SpecMuon
from optimizer_benchmark import benchmark_optimizers, benchmark_optimizers_gpt2xl
import re
import copy
import torch
import matplotlib.pyplot as plt
device = "cuda:0"

model_name = "openai-community/gpt2-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token


#I'm just using the KJV bible as a sample finetuning text. I couldn't find any canonical corpus to do this on so I thought the bible would be fine.  
with open("kjv.txt", "r") as f:

    bible = f.read()

    # remove verse numbers and punctuation
    bible = re.sub(r"^.*?\d+:\d+\s*", "", bible, flags=re.MULTILINE)
    bible = re.sub(r"\[", "", bible)
    bible = re.sub(r"\]", "", bible)
    bible = re.sub(r"\,", "", bible)
    bible = re.sub(r"\;", "", bible)

with open("romeo_and_juliet.txt", "r") as f:
    romeo_and_juliet = f.read()

# tokens = tokenizer(romeo_and_juliet[:len(romeo_and_juliet)//2],  return_tensors="pt")
tokens = tokenizer(romeo_and_juliet,  return_tensors="pt")
frspecmuon_final_losses = []
adamw_final_losses = []
for i in range(3):
        device = "cuda:0"

        model_frspecmuon = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )
        # Turns the linear layers of the model into LoRAed versions
        riemannize_experimental(model_frspecmuon, 60, exclusions=[model_frspecmuon.lm_head])

        
        model_adamw = copy.deepcopy(model_frspecmuon)


        frspecmuon = FrSpecMuon(
            model_frspecmuon,
            lr=3e-4,
            betas = (0.9,0.95),
            weight_decay = 0.01
        )

        adamw = torch.optim.AdamW(
            model_adamw.parameters(),
            lr=3e-4,
            betas = (0.9,0.95)
        )    



        final_losses = benchmark_optimizers_gpt2xl([
            {"optimizer": frspecmuon, "label": "FrSpecMuon", "model": model_frspecmuon, "uses_closure": True}, 
            {"optimizer": adamw, "label": "AdamW", "model": model_adamw, "uses_closure": True}
            ],
            tokens,
            epoch_count = 20,
            steps_per_epoch = 50,
            block_size = 1024,
            graph = True,
            tag = "bible_"+ str(i)
        )
        frspecmuon_final_losses.append(final_losses[0].item())
        adamw_final_losses.append(final_losses[1].item())

print(f"Final losses FrSpecmuon: {frspecmuon_final_losses}")
print(f"Final losses AdamW: {adamw_final_losses}")

runs = range(1, len(frspecmuon_final_losses) + 1)

plt.figure(figsize=(8, 5))

plt.plot(runs, frspecmuon_final_losses, marker='o', label='FrSpecMuon')
plt.plot(runs, adamw_final_losses, marker='^', label='AdamW')

plt.xlabel("Training Run")
plt.ylabel("Final Loss")
plt.title("Final Loss Across Training Runs")
plt.xticks(runs)
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.savefig(
        f"graphs/loss_across_runs_adamw_specmuon.png",
        dpi=300,
)
