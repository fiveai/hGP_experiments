from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Union

import numpy as np
import yaml
from emukit.core import ParameterSpace
from tqdm import tqdm

from models import Model


def run_experiment(
    performance_function: Callable[[np.ndarray], np.ndarray],
    model: Model,
    parameter_space: ParameterSpace,
    save_dir: Path,
    cov_threshold: float = 0.1,
    misclassification_threshold: float = 0.02,
    n_mc: int = 5 * 10**3,
    n_doe: int = 12,
    max_iterations: int = 150,
    metric_fns: dict = None,
    true_pf: float = None,
    seed: int = None,
) -> List[Dict[str, float]]:
    save_dir.mkdir(parents=True, exist_ok=True)
    if seed is not None:
        np.random.seed(seed)
    proposal_x = parameter_space.sample_uniform(n_mc)
    doe_idx = np.random.choice(n_mc, n_doe, replace=False)
    train_x = proposal_x[doe_idx, ...]
    proposal_x = np.delete(proposal_x, doe_idx, axis=0)

    metric_results: List[Dict[str, float]] = []

    for idx in tqdm(range(max_iterations)):
        model.train(train_x, performance_function(train_x))
        # Will repeatedly query the performance function - but this doesn't matter because only lightweight performance
        # functions are used for the experiments
        model.plot(save_dir / f"model_idx={idx}")
        if metric_fns is not None:
            metric_results.append({name: fn(model) for name, fn in metric_fns.items()})

        # compute max misclassification
        p_misclassification = model.misclassification_probability(proposal_x)
        max_misclassification = np.max(p_misclassification)
        metric_results[-1]["Maximum predicted misclassification probability"] = max_misclassification
        metric_results[-1]["n training points"] = len(train_x)

        all_x = np.concatenate([proposal_x, train_x])
        pf = model.failure_probability(all_x)
        sigma = np.sqrt(pf * (1 - pf) / len(all_x))
        if pf > 0:
            cov = sigma / pf
        else:
            cov = 1
        metric_results[-1]["p_f"] = pf
        if true_pf is not None:
            metric_results[-1]["Failure Probability Absolute Error"] = np.abs(pf - true_pf)
        metric_results[-1]["CoV"] = cov

        if max_misclassification < misclassification_threshold:
            if cov < cov_threshold:
                break
            else:
                new_samples = parameter_space.sample_uniform(n_mc)
                proposal_x = np.concatenate([proposal_x, new_samples])
        else:
            next_idx = np.argmax(p_misclassification)
            next_x = proposal_x[next_idx]
            proposal_x = np.delete(proposal_x, next_idx, axis=0)
            train_x = np.concatenate(
                [
                    train_x,
                    [
                        next_x,
                    ],
                ]
            )

    metric_results = prune_no_added_iterations(metric_results)

    with open(save_dir / "results.yaml", "w") as f:
        yaml.dump(convert_all_numpy_to_float(metric_results), f)

    return metric_results


def convert_all_numpy_to_float(to_convert: Any) -> Union[Dict, float, List]:
    """Converts all types to yaml exportables by removing numpy arrays in a nested fashion"""
    if type(to_convert) == dict:
        return {key: convert_all_numpy_to_float(value) for key, value in to_convert.items()}
    elif type(to_convert) == list:
        return [convert_all_numpy_to_float(thing) for thing in to_convert]
    elif isinstance(to_convert, np.floating):
        return to_convert.item()
    elif isinstance(to_convert, np.int64):
        return to_convert.item()
    elif isinstance(to_convert, float):
        return to_convert
    elif isinstance(to_convert, int):
        return to_convert
    else:
        raise ValueError(f"Can't convert to numpy: {to_convert} type {type(to_convert)}")


def prune_no_added_iterations(metric_results: List[Dict[str, float]]) -> List[Dict[str, float]]:
    out_list = []
    for idx, item in enumerate(metric_results):
        if idx > 0 and item["n training points"] == metric_results[idx - 1]["n training points"]:
            continue
        out_list.append(item)
    return out_list
