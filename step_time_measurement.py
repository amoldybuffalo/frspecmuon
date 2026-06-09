import gc
import re
import time

import torch
from torch.optim import AdamW

from tinygpt import TinyGPT2, Tokenizer
from riemann_layers import riemannize
from r_optimizer import FrSpecMuon


device = "cuda:0"
tokenizer = Tokenizer()


def load_bible_tokens():
    with open("kjv.txt", "r") as f:
        bible = f.read()

    bible = re.sub(r'^.*?\d+:\d+\s*', '', bible, flags=re.MULTILINE)
    bible = re.sub(r'\[', '', bible)
    bible = re.sub(r'\]', '', bible)
    bible = re.sub(r'\,', '', bible)
    bible = re.sub(r'\;', '', bible)

    return torch.tensor(
        tokenizer.encode(bible),
        dtype=torch.long,
        device=device
    )


bible_tokens = load_bible_tokens()


def benchmark(name, model, optimizer):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    step_times = []
    mem_samples = []

    for epoch in range(1):
        for i in range(512, len(bible_tokens) - 512, 1024):

            def closure():
                logits, loss, _ = model(
                    bible_tokens[i - 512:i][None, :],
                    bible_tokens[i:i + 512][None, :]
                )

                loss.backward()
                return loss

            torch.cuda.synchronize()
            start = time.perf_counter()

            loss = optimizer.step(closure)

            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start

            step_times.append(elapsed)

            # active tensor memory
            mem_samples.append(torch.cuda.memory_allocated())

            print(
                f"{name}: "
                f"{1000 * sum(step_times) / len(step_times):.2f} ms/step  ",
                end="\r"
            )

            optimizer.zero_grad()

    print()

    avg_step_ms = 1000 * sum(step_times) / len(step_times)
    avg_mem_mb = sum(mem_samples) / len(mem_samples) / 1024**2
    peak_mem_mb = torch.cuda.max_memory_allocated() / 1024**2

    print(f"\n{name} Results")
    print(f"Average step time : {avg_step_ms:.2f} ms")
    print(f"Average memory    : {avg_mem_mb:.2f} MB")
    print(f"Peak memory       : {peak_mem_mb:.2f} MB")

    return avg_step_ms, avg_mem_mb, peak_mem_mb


# ==========================================================
# FrSpecMuon
# ==========================================================

model = TinyGPT2.from_pretrained(
    "tinygpt2_ckpt_2026_02_18_20_42.pth"
).to(device)

riemannize(model, 10, exclusions=[model.lm_head])

optimizer = FrSpecMuon(model)

fr_results = benchmark("FrSpecMuon", model, optimizer)


# ==========================================================
# Cleanup
# ==========================================================

del optimizer
del model

gc.collect()
torch.cuda.empty_cache()


# ==========================================================
# AdamW
# ==========================================================

model = TinyGPT2.from_pretrained(
    "tinygpt2_ckpt_2026_02_18_20_42.pth"
).to(device)

optimizer = AdamW(model.parameters(), lr=1e-3)

adam_results = benchmark("AdamW", model, optimizer)


print("\n======================")
print("Comparison")
print("======================")
print(
    f"Step time ratio : "
    f"{fr_results[0] / adam_results[0]:.2f}x"
)
print(
    f"Memory ratio    : "
    f"{fr_results[1] / adam_results[1]:.2f}x"
)
