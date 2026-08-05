from tinygpt import TinyGPT2, TinyGPT2Config, Tokenizer, GPTLanguageModel
from matplotlib import pyplot as plt
import torch
import copy

def move_cursor_up_n_lines(n):
    print(f"\033[{n}A", end="\r")







def benchmark_optimizers(
    base_model,
    optimizer_configs,
    trainloader,
    loss_fn,
    *,
    device="cuda",
    epoch_count=20,
    graph=False,
    graph_type="epochs",
    logarithmic=False,
    smooth_size = 10,
    graph_output_dir = "graphs",
    tag="",
):
    """
    Benchmarks multiple optimizers on identical batches.

    Parameters
    ----------
    base_model : nn.Module
        Untrained model.

    optimizer_configs : list of dicts
        Each dict should contain:

            {
                "label": str,
                "optimizer_fn": callable(model) -> optimizer,
                "uses_closure": bool
            }

    trainloader : DataLoader

    loss_fn : callable(model, batch) -> Tensor
        Computes the loss for one batch.
    """

    models = [copy.deepcopy(base_model) for _ in optimizer_configs]

    for model in models:
        model.to(device)
        model.train()

    optimizers = [
        cfg["optimizer_fn"](model, cfg["hyperparameters"])
        for cfg, model in zip(optimizer_configs, models)
    ]

    labels = [cfg["label"] for cfg in optimizer_configs]
    uses_closure = [cfg["uses_closure"] for cfg in optimizer_configs]

    losses_per_step = [[] for _ in optimizers]
    avg_losses_epoch = [[] for _ in optimizers]

    for epoch in range(epoch_count):

        total_losses = torch.zeros(len(optimizers))

        print(f"\nEpoch {epoch + 1}")

        for step, batch in enumerate(trainloader):

            losses = torch.zeros(len(optimizers))

            for i, (model, optimizer) in enumerate(zip(models, optimizers)):

                if uses_closure[i]:

                    def closure(backwards_pass = True):
                        
                        optimizer.zero_grad()
                        loss = loss_fn(model, batch)

                        if backwards_pass:
                            loss.backward()
                        return loss

                    loss = optimizer.step(closure).item()
                   

                else:

                    optimizer.zero_grad()

                    loss = loss_fn(model, batch)
                    loss.backward()

                    loss = loss.item()

                    optimizer.step()

                losses[i] = loss
                losses_per_step[i].append(loss)

            total_losses += losses
            avg_losses = total_losses / (step + 1)

            print(f"progress: {100 * (step + 1) /  len(trainloader):0.2f}%")

            for i in range(len(optimizers)):
                print(
                    f"{labels[i]} loss this step: "
                    f"{losses[i]:0.3f}, "
                    f"avg loss (epoch): {avg_losses[i]:0.3f}"
                )

            move_cursor_up_n_lines(len(optimizers) + 1)


        for i in range(len(optimizers)):
            avg_losses_epoch[i].append(avg_losses[i])

        print((len(optimizers) + 2) * "\n")
            
    if graph:

        fig, ax = plt.subplots(figsize=(8, 5))

        if logarithmic:
            ax.set_yscale("log")

        if graph_type == "epochs":

            for i, label in enumerate(labels):

                ax.plot(
                    range(1, len(avg_losses_epoch[i])+1),
                    avg_losses_epoch[i],
                    label=label,
                )

            ax.set_xlabel("Epoch")
            ax.set_ylabel("Average Loss")

        elif graph_type == "steps":

            for i, label in enumerate(labels):

                ax.plot(
                    losses_per_step[i],
                    label=label,
                )

            ax.set_xlabel("Step")
            ax.set_ylabel("Loss")
        
        elif graph_type == "steps_smoothed":


            for i, label in enumerate(labels):

                losses = losses_per_step[i]

                smoothed = [
                    sum(losses[j:j + smooth_size]) / len(losses[j:j + smooth_size])
                    for j in range(0, len(losses), smooth_size)
                ]

                ax.plot(
                    range(1, len(smoothed) + 1),
                    smoothed,
                    linewidth=2,
                    label=label,
                )

            ax.set_xlabel(f"Step")
            ax.set_ylabel("Loss ({smooth_size}-step average)")

        elif graph_type == "EMA":
            
            for i, label in enumerate(labels):
                alpha = 0.02  # smaller = smoother
                ema = np.empty_like(losses_per_step[i])
                ema[0] = losses_per_step[i][0]
            
                for j in range(1, len(losses_per_step[i])):
                    ema[j] = alpha * losses_per_step[i][j] + (1 - alpha) * ema[j - 1]

                ax.plot(
                    range(1, len(ema) + 1),
                    ema,
                    linewidth=2,
                    label=label,
                )
            
            
            ax.set_xlabel(f"Step")
            ax.set_ylabel("Loss (Exponential Moving Average)")

            



        ax.legend()
        ax.grid(True)
        plt.tight_layout()

       
        plot_title = ""

        for i in range(len(optimizers)):
            plot_title += labels[i]

            if i + 1 < len(optimizers):
                plot_title += " vs "

        plt.title(plot_title)
        plt.grid(True)
        plt.legend()

        plt.tight_layout()

        plt.savefig(
            f"{graph_output_dir}/{plot_title}{('_' + tag) if tag else ''}{"_log_plot" if logarithmic else ""}.png",
            dpi=300,
        )

    return losses_per_step, avg_losses_epoch

   
