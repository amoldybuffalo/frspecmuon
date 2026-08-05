import csv
import itertools
from optimizer_benchmark import benchmark_optimizers
import numpy as np
import torch
def build_grid_configs(
    grid,
    optimizer_fn,
    *,
    uses_closure=True,
):
    keys = list(grid.keys())

    configs = []

    for values in itertools.product(*(grid[k] for k in keys)):

        hyperparameters = dict(zip(keys, values))

        configs.append({
            "label": ", ".join(
                f"{k}={v}" for k, v in hyperparameters.items()
            ),
            "optimizer_fn": optimizer_fn,
            "hyperparameters": hyperparameters,
            "uses_closure": uses_closure,
        })

    return configs


def grid_search(
    *,
    base_model,
    trainloader,
    loss_fn,
    optimizer_fn,
    grid,
    benchmark_fn=benchmark_optimizers,
    device="cuda",
    epoch_count=10,
    graph=False,
    graph_type="epochs",
    logarithmic=True,
    smooth_size=10,
    lower_is_better=True,
    output_csv=None,
    grid_batch_size = 2
):
    torch.cuda.empty_cache()
    configs = build_grid_configs(
        grid,
        optimizer_fn,
    )

    config_batches = [configs[i:i+grid_batch_size] for i in range(0, len(configs), grid_batch_size)]

    final_losses = []
    for batched_configs in config_batches:
        losses_per_step, avg_losses_epoch = benchmark_fn(
            base_model=base_model,
            optimizer_configs=batched_configs,
            trainloader=trainloader,
            loss_fn=loss_fn,
            device=device,
            epoch_count=epoch_count,
            graph=graph,
            graph_type=graph_type,
            logarithmic=logarithmic,
            smooth_size=smooth_size,
            tag="grid_search",
        )

        final_losses.extend([np.array(step_losses_per_optim).mean() for step_losses_per_optim in losses_per_step]) 

    results = []

    for config, loss in zip(configs, final_losses):

        results.append({
            "hyperparameters": config["hyperparameters"],
            "loss": loss,
            "label": config["label"],
        })

    results.sort(
        key=lambda x: x["loss"],
        reverse=not lower_is_better,
    )

    print("\nFinal ranking:\n")

    for i, result in enumerate(results, 1):

        print(
            f"{i}. "
            f"{result['hyperparameters']} "
            f"-> {result['loss']:.6f}"
        )

    if output_csv is not None:

        with open(output_csv, "w", newline="") as f:

            writer = csv.writer(f)

            writer.writerow(list(grid.keys()) + ["final_loss"])

            for result in results:

                writer.writerow(
                    [result["hyperparameters"][k] for k in grid]
                    + [result["loss"]]
                )

    return {
        "best_hyperparameters": results[0]["hyperparameters"],
        "best_loss": results[0]["loss"],
        "results": results,
    }