from tinygpt import TinyGPT2, TinyGPT2Config, Tokenizer, GPTLanguageModel
from matplotlib import pyplot as plt
import torch


def move_cursor_up_n_lines(n):
    print(f"\033[{n}A", end="\r")

def benchmark_optimizers(optimizer_configs, tokens, **kwargs):

    device = kwargs.get("device", "cuda:0")
    epoch_count = kwargs.get("epoch_count", 20)
    steps_per_epoch = kwargs.get("steps_per_epoch", 100)
    graph = kwargs.get("graph", False)
    tag = kwargs.get("tag", "")
    block_size = kwargs.get("block_size", 512)
    print(f"Tag is: {tag}")
    
    models = [config["model"] for config in optimizer_configs]
    optimizers = [config["optimizer"] for config in optimizer_configs]
    labels = [config["label"] for config in optimizer_configs]
    uses_closure = [config["uses_closure"] for config in optimizer_configs]

    for model in models:
        model.train()

    losses_per_step = []
    avg_losses_epoch = []  

    for epoch in range(epoch_count): 
        total_losses = torch.zeros(len(optimizers))
        avg_losses =  torch.zeros(len(optimizers))
        

        print(f"\nEpoch {epoch + 1}")

        for step in range(steps_per_epoch):
            losses = torch.zeros(len(optimizers))

            # Sample a random chunk of the text
            # I believe this is normal in finetuning, but correct me if I'm wrong
            token_index = torch.randint(
                block_size,
                len(tokens) - block_size,
                (),
                device=device,
            ).item()

            #Evaluate all models on the same data 
            for i in range(len(optimizers)):
                if uses_closure[i]:
                    def closure():
                        logits, loss, _ = models[i](
                            tokens[token_index - block_size:token_index][None, :],
                            tokens[token_index:token_index + block_size][None, :],
                        )

                        loss.backward()

                        return loss

                    losses[i] = optimizers[i].step(closure)

                else:
                    logits, loss, _ = models[i](
                            tokens[token_index - block_size:token_index][None, :],
                            tokens[token_index:token_index + block_size][None, :],
                    )

                    loss.backward()

                    losses[i] = loss.item()
                    optimizers[i].step()
            
            total_losses += losses
        
            avg_losses = total_losses / (step+1)

            for optimizer in optimizers:
                optimizer.zero_grad()

            losses_per_step.append(losses)
            print(f"progress: {100 * step / steps_per_epoch:0.2f}%    ")
            for i in range(len(optimizers)):
                print(f"{labels[i]} loss this step: {losses[i]:0.3f}, avg loss (epoch): {avg_losses[i]:0.3f}     ")
            move_cursor_up_n_lines(len(optimizers) + 1)
        
        print("\n" * len(optimizers))
        print(f"\nEpoch {epoch + 1} complete:")
        avg_losses_epoch.append(avg_losses)
        for i in range(len(optimizers)):
            print(f"{labels[i]} avg loss (epoch): {avg_losses_epoch[epoch][i]:0.3f} ")
        print("\n")

    if graph:
        plt.figure(figsize=(8, 5))
        for i in range(len(optimizers)):
            epoch_losses = [avg_losses_epoch[epoch][i].item() for epoch in range(epoch_count)]
            plt.plot(
                range(1, len(epoch_losses) + 1),
                epoch_losses,
                linewidth=2,
                label=labels[i],
            )


        plt.xlabel("Epoch")
        plt.ylabel("Average Loss")
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
            f"graphs/{plot_title}{("_" + tag) if (tag != "") else ""}.png",
            dpi=300,
        )

    return avg_losses_epoch[epoch_count-1] 




def benchmark_optimizers_gpt2xl(optimizer_configs, tokens, **kwargs):

    device = kwargs.get("device", "cuda:0")
    epoch_count = kwargs.get("epoch_count", 20)
    steps_per_epoch = kwargs.get("steps_per_epoch", 100)
    graph = kwargs.get("graph", False)
    tag = kwargs.get("tag", "")
    block_size = kwargs.get("block_size", 512)

    print(f"Tag is: {tag}")

    models = [config["model"] for config in optimizer_configs]
    optimizers = [config["optimizer"] for config in optimizer_configs]
    labels = [config["label"] for config in optimizer_configs]
    uses_closure = [config["uses_closure"] for config in optimizer_configs]

    for model in models:
        model.train()

        # Important for training GPT-2
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    losses_per_step = [[] for _ in range(len(optimizers))]
    avg_losses_epoch = []
    
    tokens = tokens["input_ids"][0]
    for epoch in range(epoch_count):

        total_losses = torch.zeros(len(optimizers))
        avg_losses = torch.zeros(len(optimizers))

        print(f"\nEpoch {epoch + 1}")

        for step in range(steps_per_epoch):

            losses = torch.zeros(len(optimizers))
            
            token_index = torch.randint(
                block_size,
                len(tokens),
                (),
                device=device,
            ).item()

            input_ids = tokens[
                token_index - block_size : token_index
            ][None, :].to(device)


            # Evaluate all models on same batch
            for i in range(len(optimizers)):

                if uses_closure[i]:

                    def closure():
                        optimizers[i].zero_grad()

                        outputs = models[i](
                            input_ids=input_ids,
                            labels=input_ids,
                        )
                    

                        loss = outputs.loss
                        loss.backward()
                       

                        return loss
            
                    loss = optimizers[i].step(closure).item()

                else:
                    optimizers[i].zero_grad()

                    outputs = models[i](
                        input_ids=input_ids,
                        labels=input_ids,
                    )

                    loss = outputs.loss

                    loss.backward()

                    loss = loss.item()

                    optimizers[i].step()

                losses[i] = loss

                losses_per_step[i].append(loss)

            total_losses += losses
            avg_losses = total_losses / (step + 1)


            

            print(f"progress: {100 * step / steps_per_epoch:0.2f}%")

            for i in range(len(optimizers)):
                print(
                    f"{labels[i]} loss this step: "
                    f"{losses[i]:0.3f}, "
                    f"avg loss (epoch): {avg_losses[i]:0.3f}"
                )

            move_cursor_up_n_lines(len(optimizers) + 1)

        print("\n" * len(optimizers))

        print(f"\nEpoch {epoch + 1} complete:")

        avg_losses_epoch.append(avg_losses)

        for i in range(len(optimizers)):
            print(
                f"{labels[i]} avg loss (epoch): "
                f"{avg_losses_epoch[epoch][i]:0.3f}"
            )

        print("\n")

    if graph:
        plt.figure(figsize=(8, 5))

        for i in range(len(optimizers)):

            epoch_losses = [avg_losses_epoch[epoch][i] for epoch in range(len(avg_losses_epoch))]
            plt.plot(
                range(1, len(epoch_losses) + 1),
                epoch_losses,
                linewidth=2,
                label=labels[i],
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

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
            f"graphs/{plot_title}{('_' + tag) if tag else ''}.png",
            dpi=300,
        )

    return avg_losses_epoch[-1]


def benchmark_optimizers_resnet(optimizer_configs, trainloader, criterion, **kwargs):
    device = kwargs.get("device", "cuda:0")
    epoch_count = kwargs.get("epoch_count", 20)
    steps_per_epoch = kwargs.get("steps_per_epoch", None)
    graph = kwargs.get("graph", False)
    tag = kwargs.get("tag", "")

    print(f"Tag is: {tag}")

    models = [config["model"] for config in optimizer_configs]
    optimizers = [config["optimizer"] for config in optimizer_configs]
    labels = [config["label"] for config in optimizer_configs]
    uses_closure = [config["uses_closure"] for config in optimizer_configs]

    for model in models:
        model.train()

    losses_per_step = [[] for _ in range(len(optimizers))]
    avg_losses_epoch = []

    for epoch in range(epoch_count):

        total_losses = torch.zeros(len(optimizers))

        print(f"\nEpoch {epoch + 1}")

        step = 0

        for images, targets in trainloader:

            # if steps_per_epoch is not None and step >= steps_per_epoch:
            #     break

            images = images.to(device)
            targets = targets.to(device)

            losses = torch.zeros(len(optimizers))

            # Evaluate every optimizer on the exact same batch
            for i in range(len(optimizers)):

                if uses_closure[i]:

                    def closure():
                        optimizers[i].zero_grad()

                        outputs = models[i](images)
                        loss = criterion(outputs, targets)

                        loss.backward()

                        return loss

                    loss = optimizers[i].step(closure).item()

                else:

                    optimizers[i].zero_grad()

                    outputs = models[i](images)
                    loss = criterion(outputs, targets)

                    loss.backward()

                    loss = loss.item()

                    optimizers[i].step()

                losses[i] = loss
                losses_per_step[i].append(loss)

            total_losses += losses
            avg_losses = total_losses / (step + 1)

            print(f"progress: {100 * (step + 1) / (steps_per_epoch if steps_per_epoch is not None else len(trainloader)):0.2f}%")

            for i in range(len(optimizers)):
                print(
                    f"{labels[i]} loss this step: "
                    f"{losses[i]:0.3f}, "
                    f"avg loss (epoch): {avg_losses[i]:0.3f}"
                )

            move_cursor_up_n_lines(len(optimizers) + 1)

            step += 1

        print("\n" * len(optimizers))

        print(f"\nEpoch {epoch + 1} complete:")

        avg_losses_epoch.append(avg_losses.clone())

        for i in range(len(optimizers)):
            print(
                f"{labels[i]} avg loss (epoch): "
                f"{avg_losses_epoch[-1][i]:0.3f}"
            )

        print()

    if graph:
        plt.figure(figsize=(8, 5))

        for i in range(len(optimizers)):
            epoch_losses = [epoch[i] for epoch in avg_losses_epoch]

            plt.plot(
                range(1, len(epoch_losses) + 1),
                epoch_losses,
                linewidth=2,
                label=labels[i],
            )

        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        plot_title = " vs ".join(labels)

        plt.title(plot_title)
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        plt.savefig(
            f"graphs/{plot_title}{('_' + tag) if tag else ''}.png",
            dpi=300,
        )

    return avg_losses_epoch[-1]

   
