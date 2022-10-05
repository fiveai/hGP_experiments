from pathlib import Path
from typing import Callable
from typing import Dict
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from emukit.core import ParameterSpace

asymptote_at_1 = [
    "Average Precision",
    "f1",
]
asymptote_at_0 = ["CoV", "p_f", "Maximum predicted misclassification probability", "Failure Probability Absolute Error"]
no_asymptote = ["n training points"]

colours = ["blue", "red", "green", "black", "yellow"]


def plot_metrics(metric_dicts: Dict[str, List[List[Dict[str, float]]]], save_dir: Path, pf: float):
    model_names = list(metric_dicts.keys())
    metric_names = list(metric_dicts[model_names[0]][0][0].keys())
    n_repeats = len(metric_dicts[model_names[0]])

    for name in metric_names:
        plt.clf()
        largest_upper = 0.0
        lowest_low = 1.0
        for idx, model_name in enumerate(model_names):

            n_iters = max(len(repeat) for repeat in metric_dicts[model_name])
            result_repeats_series = np.full((n_repeats, n_iters), np.nan)

            for idx_1, repeat in enumerate(metric_dicts[model_name]):
                for idx_2, iteration in enumerate(repeat):
                    result_repeats_series[idx_1, idx_2] = iteration[name]
            plt.plot(
                np.nanmean(result_repeats_series, axis=0),
                label=model_name,
                color=colours[idx],
            )
            lower = np.nanmin(result_repeats_series, axis=0)
            upper = np.nanmax(result_repeats_series, axis=0)
            if name in asymptote_at_1 or name in asymptote_at_0:
                eps = 1e-9
                lower = np.clip(lower, a_min=eps, a_max=1 - eps)
                upper = np.clip(upper, a_min=eps, a_max=1 - eps)
                largest_upper = 1 - 1e-8
                lowest_low = np.min(np.concatenate([[lowest_low], lower[lower > eps]])).item()

            plt.fill_between(
                range(n_iters),
                lower,
                upper,
                alpha=0.3,
                # label=model_name,
                color=colours[idx],
            )

        if name in asymptote_at_1:
            plt.yscale("logit")
            plt.ylim([0.5, largest_upper])
        elif name in asymptote_at_0:
            plt.yscale("log")
            plt.ylim([lowest_low, largest_upper])
        elif name in no_asymptote:
            ...
        else:
            raise ValueError(f"Don't know where to place asymptote for {name}")

        if name == "p_f":
            plt.plot([0, n_iters], [pf, pf])

        plt.xlim([0, n_iters])
        plt.ylabel(name)
        plt.xlabel("Iterations")
        plt.legend()
        plt.tight_layout()
        plt.savefig(save_dir / f"{name}.pdf")


def plot_ground_truth(
    performance: Callable[[np.ndarray], np.ndarray],
    parameter_space: ParameterSpace,
    save_dir: Path,
):
    n_parameters = len(parameter_space.parameter_names)
    if n_parameters == 1:

        x_samples = np.sort(parameter_space.sample_uniform(10**6), axis=0)
        plt.plot(x_samples, performance(x_samples), label="Real valued")
        plt.plot(x_samples, ~np.isfinite(performance(x_samples)), label="1(y=NaN)")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()

    elif n_parameters == 2:
        bounds = parameter_space.get_bounds()
        x = np.linspace(bounds[0][0], bounds[0][1], 100)
        y = np.linspace(bounds[0][0], bounds[0][1], 100)

        X, Y = np.meshgrid(x, y)
        Z = performance(np.stack([X, Y], axis=-1))[..., 0]

        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")
        _ = ax.plot_surface(X, Y, Z, label="Real valued")
        tmp = np.ones_like(Z)
        tmp[~np.isnan(Z)] = np.nan
        _ = ax.plot_surface(X, Y, tmp, label="NaN")
        ax.set_xlabel("Adversary start x")
        ax.set_ylabel("Adversary velocity (constant)")
        ax.set_zlabel("Rule robustness")

    else:
        print(f"Don't know how to plot {n_parameters}")
    plt.savefig(save_dir / "gt.pdf", bbox_inches="tight")
