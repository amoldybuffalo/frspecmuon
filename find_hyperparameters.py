"""
Hyperparameter search harness for benchmark_optimizers_resnet, wired to the
real FrSpecMuon / riemannized ResNet50 / ImageNetV2 setup.

CONFIRMED: benchmark_optimizers_resnet is expected to return
(losses_per_step, avg_losses_epoch):
  - losses_per_step: list of length num_optimizers, each a list of raw
    per-step losses for that config (in optimizer_configs order).
  - avg_losses_epoch: list of length epoch_count, each a tensor of shape
    [num_optimizers] (one avg loss per config, in optimizer_configs order).
This requires a one-line change to your real function -- it already builds
losses_per_step internally, just add it to the return statement:
    return losses_per_step, avg_losses_epoch
Per-step is used for pruning resolution (epoch_count is small during search,
so per-epoch alone gives the pruner too few checkpoints); per-epoch is used
for the final tail-averaged score, since it's already a smoothed signal.

CORRECTNESS NOTE: benchmark_optimizers_resnet trains models[0] == the
`base_model` arg you pass it *in place* (it only deep-copies for configs
after index 0). This script already deep-copies BASE_MODEL into a
disposable `fresh_model` before every call, so BASE_MODEL itself is never
touched -- but don't call benchmark_optimizers_resnet elsewhere with
BASE_MODEL directly, or it'll get trained as a side effect.

OPEN QUESTION (search for "RANK NOTE" below): riemannize_experimental(model,
20, ...) bakes rank=20 into BASE_MODEL's architecture once, before the
search starts. If you want to sweep rank as a search dimension too, that's
a structurally different (and much more expensive) loop than sweeping lr /
relaxation_tolerance -- see the note.
"""

import copy
import statistics
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from torchvision.datasets import ImageNet
from imagenetv2_pytorch import ImageNetV2Dataset
from frspecmuon import FrSpecMuon
from riemann_layers import riemannize, riemannize_experimental, split_parameters
from optimizer_benchmark import benchmark_optimizers_resnet
from muon import SingleDeviceMuonWithAuxAdam  # not used in this search yet -- kept for a future baseline comparison

import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

TAIL_FRACTION = 0.2           # fraction of epochs (not steps) to average over -- epoch counts are small here
COARSE_EPOCHS = 3             # short budget for the coarse search
REFINE_EPOCHS = 6
FINAL_EPOCHS = 10             # matches your example's epoch_count=10
COARSE_N_TRIALS = 15
REFINE_N_TRIALS = 5
TOP_K_FOR_FINAL = 3
FINAL_SEEDS = [0, 1, 2]

def init_frspecmuon(optimizer, model, hyperparameters):
    return optimizer(model, **hyperparameters)

def init_muon(optimizer, model, hyperparameters):
    riemann_params, other_params = split_parameters(model)
    hyperparameters["lr"] = hyperparameters.get("lr", 0.01)
    hyperparameters["weight_decay"] = hyperparameters.get("weight_decay", 0.01)
    hyperparameters["momentum"] = hyperparameters.get("momentum", 0.95)
    hyperparameters["lr"] = hyperparameters.get("lr", 0.01)
    param_groups = [
        dict(params=riemann_params, use_muon=True,
                lr=hyperparameters["lr"], weight_decay=hyperparameters["weight_decay"], momentum=hyperparameters["momentum"]),
        dict(params=other_params, use_muon=False,
                lr=3e-4, betas=(0.9, 0.999), weight_decay=0.01),
    ]

    return optimizer(param_groups)


def init_adamw(optimizer, model, hyperparameters):
    return optimizer(model.parameters(), **hyperparameters)

OPTIMIZER = FrSpecMuon
OPTIMIZER_INIT_FUNCTION = init_frspecmuon
OPTIMIZER_NAME = "FrSpecMuon"
DEVICE = "cuda:0"
TRAIN_DATASET = 


def build_base_model():
    """
    Builds the pretrained + riemannized starting point exactly once. This
    is NOT re-run per trial -- every trial deep-copies the result. If
    riemannize_experimental returns a new model instead of mutating in
    place, change the last line to `model = riemannize_experimental(...)`.

    RANK NOTE: the `20` here is baked into the model's architecture at
    construction time -- it is not an optimizer hyperparameter the way lr
    and relaxation_tolerance are. If you want rank in the search space,
    run_single_trial would need to call build_base_model(rank=...) fresh
    per trial (rebuilding + re-riemannizing from the pretrained checkpoint)
    instead of deep-copying the single shared BASE_MODEL -- meaningfully
    more expensive per trial. Left as a fixed constant for now; say the
    word if you want it swept.
    """
    weights = models.ResNet50_Weights.IMAGENET1K_V1
    model = models.resnet50(weights=weights).to(DEVICE)

    riemannize_experimental(model, 20, exclusions=[model.fc])
    return model, weights





def build_trainloader(transform):
    
    return DataLoader(
        TRAIN_DATASET,
        batch_size=256,
        shuffle=True,
        num_workers=15,
        pin_memory=True,
    )


BASE_MODEL, _weights = build_base_model()
TRAINLOADER = build_trainloader(_weights.transforms())
CRITERION = nn.CrossEntropyLoss()


def score_from_losses(losses: list[float]) -> float:
    """Mean loss over the final TAIL_FRACTION of epochs. Lower is better."""
    if not losses:
        raise ValueError("No losses returned for this config")
    tail_len = max(1, int(len(losses) * TAIL_FRACTION))
    return statistics.mean(losses[-tail_len:])


def extract_losses(result, index: int = 0) -> tuple[list[float], list[float]]:
    """
    result == (losses_per_step, avg_losses_epoch). Since run_single_trial
    always passes a single-config list, index is always 0 there.
    Returns (per_step_losses, per_epoch_losses) for that config.
    """
    losses_per_step, avg_losses_epoch = result
    per_step = list(losses_per_step[index])
    per_epoch = [epoch_tensor[index].item() for epoch_tensor in avg_losses_epoch]
    return per_step, per_epoch


@dataclass
class SearchSpace:
    # (low, high, log) for continuous params; list for categorical.
    # Only lr and relaxation_tolerance are shown in your example -- add any
    # other keys FrSpecMuon's hyperparameters dict actually accepts.
    ranges: dict = field(default_factory=lambda: {
        "lr": (0.02, 0.02, True),
        "q_multiplier": (0.5, 2, True)
    })

    def suggest(self, trial: optuna.Trial) -> dict:
        hparams = {}
        for name, spec in self.ranges.items():
            if isinstance(spec, list):
                hparams[name] = trial.suggest_categorical(name, spec)
            else:
                low, high, log = spec
                hparams[name] = trial.suggest_float(name, low, high, log=log)
        return hparams

    def narrowed_around(self, best_params: dict, factor: float = 3.0) -> "SearchSpace":
        new_ranges = {}
        for name, spec in self.ranges.items():
            if isinstance(spec, list):
                new_ranges[name] = spec
                continue
            low, high, log = spec
            center = best_params[name]
            if log:
                lo = max(low, center / factor)
                hi = min(high, center * factor)
            else:
                span = (high - low) / factor
                lo = max(low, center - span)
                hi = min(high, center + span)
            new_ranges[name] = (lo, hi, log)
        ns = SearchSpace()
        ns.ranges = new_ranges
        return ns


def build_config(hparams: dict, label: str) -> dict:
    """Matches the optimizer_configs entry format from your snippet."""
    return {
        "optimizer": OPTIMIZER,
        "hyperparameters": hparams,
        "label": label,
        "init_function": OPTIMIZER_INIT_FUNCTION,
        "uses_closure": True,
    }


def run_single_trial(hparams: dict, epoch_count: int,
                      trial: optuna.Trial | None = None,
                      tag_prefix: str = "search") -> float:
    label = f"trial_{trial.number}" if trial is not None else "eval"
    config = build_config(hparams, label)

    fresh_model = copy.deepcopy(BASE_MODEL)  # never train BASE_MODEL itself

    result = benchmark_optimizers_resnet(
        fresh_model,
        [config],
        TRAINLOADER,
        CRITERION,
        device=DEVICE,
        epoch_count=epoch_count,
        graph=False,          # no need to plot during search -- only for the final run
        logarithmic=False,
        tag=f"{tag_prefix}_{label}",
    )
    per_step_losses, per_epoch_losses = extract_losses(result, index=0)

    if trial is not None:
        # Report per-step (not per-epoch) so the pruner has enough
        # resolution to kill a bad trial partway through epoch 1, instead
        # of waiting for a full epoch_count of only 2-3 checkpoints.
        # Use a running tail-average as the reported value so early noisy
        # single-step losses don't trigger spurious pruning.
        report_every = max(1, len(per_step_losses) // 50)  # cap at ~50 reports
        for step_idx in range(0, len(per_step_losses), report_every):
            partial_score = score_from_losses(per_step_losses[: step_idx + 1])
            trial.report(partial_score, step=step_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Final score still uses the smoothed epoch-level series, not raw steps.
    return score_from_losses(per_epoch_losses)


def make_objective(space: SearchSpace, epoch_count: int,
                    tag_prefix: str) -> Callable[[optuna.Trial], float]:
    def objective(trial: optuna.Trial) -> float:
        hparams = space.suggest(trial)
        return run_single_trial(hparams, epoch_count, trial=trial, tag_prefix=tag_prefix)
    return objective


def run_search(space: SearchSpace, n_trials: int, epoch_count: int,
                seed: int = 0, study_name: str = "search") -> optuna.Study:
    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        sampler=TPESampler(seed=seed, multivariate=True),
        # n_warmup_steps is now in raw per-step report indices (see
        # run_single_trial's report_every) rather than epochs -- pruning is
        # disabled until the 2nd checkpoint report so the very first noisy
        # single-batch loss can't trigger a spurious prune.
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=1),
    )
    study.optimize(make_objective(space, epoch_count, study_name), n_trials=n_trials)
    return study


def top_k_trials(study: optuna.Study, k: int) -> list[optuna.trial.FrozenTrial]:
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return sorted(completed, key=lambda t: t.value)[:k]


def final_comparison(candidates: list[dict], seeds: list[int],
                      epoch_count: int) -> list[dict]:
    """
    Re-runs the top-K candidates across multiple seeds individually (so we
    get a clean mean/std per config), THEN makes one more call with all
    top-K configs together and graph=True so you get the built-in comparison
    plot from benchmark_optimizers_resnet for the writeup.
    """
    per_config_scores = {i: [] for i in range(len(candidates))}
    for seed in seeds:
        for i, hparams in enumerate(candidates):
            score = run_single_trial(hparams, epoch_count, tag_prefix=f"final_seed{seed}")
            per_config_scores[i].append(score)

    results = []
    for i, hparams in enumerate(candidates):
        scores = per_config_scores[i]
        results.append({
            "hparams": hparams,
            "mean": statistics.mean(scores),
            "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
            "scores": scores,
        })
    results.sort(key=lambda r: r["mean"])

    # Final side-by-side plot using your existing visualization, single seed.
    labeled_configs = [
        build_config(r["hparams"], f"rank{rank+1} ({r['mean']:.4f} +- {r['std']:.4f})")
        for rank, r in enumerate(results)
    ]
    fresh_model = copy.deepcopy(BASE_MODEL)
    benchmark_optimizers_resnet(
        fresh_model,
        labeled_configs,
        TRAINLOADER,
        CRITERION,
        device=DEVICE,
        epoch_count=epoch_count,
        graph=True,
        logarithmic=True,
        tag="final_comparison",
    )

    return results


def main():
    assert BASE_MODEL is not None, "BASE_MODEL failed to build"
    assert TRAINLOADER is not None, "Set TRAINLOADER"
    assert CRITERION is not None, "Set CRITERION"

    space = SearchSpace()

    print(f"=== Coarse search: {COARSE_N_TRIALS} trials, {COARSE_EPOCHS} epochs each ===")
    coarse_study = run_search(space, COARSE_N_TRIALS, COARSE_EPOCHS, study_name="coarse")
    coarse_best = coarse_study.best_params
    print("Coarse best:", coarse_best, "score:", coarse_study.best_value)

    print(f"\n=== Refinement search: {REFINE_N_TRIALS} trials, {REFINE_EPOCHS} epochs each ===")
    refined_space = space.narrowed_around(coarse_best, factor=3.0)
    refine_study = run_search(refined_space, REFINE_N_TRIALS, REFINE_EPOCHS, study_name="refine")
    print("Refined best:", refine_study.best_params, "score:", refine_study.best_value)

    print(f"\n=== Final multi-seed check on top {TOP_K_FOR_FINAL} refined candidates ===")
    top_trials = top_k_trials(refine_study, TOP_K_FOR_FINAL)
    candidates = [t.params for t in top_trials]
    final_results = final_comparison(candidates, FINAL_SEEDS, FINAL_EPOCHS)

    print("\nFinal ranking (mean +/- std over seeds", FINAL_SEEDS, "):")
    for rank, r in enumerate(final_results, 1):
        print(f"{rank}. score={r['mean']:.6f} +/- {r['std']:.6f}  hparams={r['hparams']}")

    return final_results


if __name__ == "__main__":
    main()