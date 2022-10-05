import datetime
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import yaml
from emukit.core import ContinuousParameter
from emukit.core import ParameterSpace
from joblib import Parallel
from joblib import delayed

from active_learning import convert_all_numpy_to_float
from active_learning import run_experiment
from metrics import avg_precision
from metrics import create_f1_evaluator
from models import Hierarchical
from models import MaskedGPClassifier
from models import MaskedModel
from plots import plot_ground_truth
from plots import plot_metrics
from test_functions import performance_function
from utils import summarise_results


def main(
    misclassification_threshold: float = 0.02,
    n_repeats: int = 5,
    save_dir: Path = Path("tmp"),
    n_jobs: int = 15,
    max_iter: int = 150,
):
    np.random.seed(0)
    random_state = np.random.randint(np.iinfo(np.int32).max, size=5 * n_repeats)
    save_dir = save_dir / datetime.datetime.now().strftime("%Y_%B_%d_%p%I:%M")
    save_dir = save_dir / "test"
    save_dir.mkdir(parents=True, exist_ok=True)

    parameter_space = ParameterSpace([ContinuousParameter("x1", 0, 1)])

    metric_fns = {
        "f1": create_f1_evaluator(parameter_space, performance_function),
        "Average Precision": avg_precision(parameter_space, performance_function),
    }

    (save_dir).mkdir(parents=True, exist_ok=True)

    plot_ground_truth(performance_function, parameter_space, save_dir)

    analytic_failure_probability = 0.036906

    all_results = Parallel(n_jobs=n_jobs)(
        list(
            (
                *(
                    delayed(run_experiment)(
                        performance_function=performance_function,
                        model=Hierarchical(),
                        parameter_space=parameter_space,
                        save_dir=save_dir / f"hierarchical_{idx}",
                        metric_fns=metric_fns,
                        misclassification_threshold=misclassification_threshold,
                        true_pf=analytic_failure_probability,
                        seed=random_state[idx],
                        max_iterations=max_iter,
                    )
                    for idx in range(n_repeats)
                ),
                *(
                    delayed(run_experiment)(
                        performance_function=performance_function,
                        model=MaskedGPClassifier(),
                        parameter_space=parameter_space,
                        save_dir=save_dir / f"masked_gp_cls_{idx}",
                        metric_fns=metric_fns,
                        misclassification_threshold=misclassification_threshold,
                        true_pf=analytic_failure_probability,
                        seed=random_state[n_repeats + idx],
                        max_iterations=max_iter,
                    )
                    for idx in range(n_repeats)
                ),
                *(
                    delayed(run_experiment)(
                        performance_function=performance_function,
                        model=MaskedModel(mask_with=1.0),
                        parameter_space=parameter_space,
                        save_dir=save_dir / f"masked_{idx}",
                        metric_fns=metric_fns,
                        misclassification_threshold=misclassification_threshold,
                        true_pf=analytic_failure_probability,
                        seed=random_state[2 * n_repeats + idx],
                        max_iterations=max_iter,
                    )
                    for idx in range(n_repeats)
                ),
                *(
                    delayed(run_experiment)(
                        performance_function=performance_function,
                        model=MaskedModel(mask_with=0.5),
                        parameter_space=parameter_space,
                        save_dir=save_dir / f"masked05_{idx}",
                        metric_fns=metric_fns,
                        misclassification_threshold=misclassification_threshold,
                        true_pf=analytic_failure_probability,
                        seed=random_state[3 * n_repeats + idx],
                        max_iterations=max_iter,
                    )
                    for idx in range(n_repeats)
                ),
                *(
                    delayed(run_experiment)(
                        performance_function=performance_function,
                        model=MaskedModel(mask_with=0.1),
                        parameter_space=parameter_space,
                        save_dir=save_dir / f"masked01_{idx}",
                        metric_fns=metric_fns,
                        misclassification_threshold=misclassification_threshold,
                        true_pf=analytic_failure_probability,
                        seed=random_state[4 * n_repeats + idx],
                        max_iterations=max_iter,
                    )
                    for idx in range(n_repeats)
                ),
            )
        )
    )

    metric_results_hierarchical = all_results[:n_repeats]
    metric_results_masked_gpcls = all_results[n_repeats : 2 * n_repeats]
    metric_results_masked = all_results[2 * n_repeats : 3 * n_repeats]
    metric_results_masked05 = all_results[3 * n_repeats : 4 * n_repeats]
    metric_results_masked01 = all_results[4 * n_repeats : 5 * n_repeats]

    result_dict: Dict[str, List[List[Dict[str, float]]]] = {
        "Hierarchical GP": metric_results_hierarchical,
        r"Masked GP $\alpha=1$": metric_results_masked,
        r"Masked GP $\alpha=0.5$": metric_results_masked05,
        r"Masked GP $\alpha=0.1$": metric_results_masked01,
        "Masked GP Classification": metric_results_masked_gpcls,
    }

    plot_metrics(
        result_dict,
        save_dir,
        analytic_failure_probability,
    )
    with open(save_dir / "results.yaml", "w") as f:
        yaml.dump(convert_all_numpy_to_float(summarise_results(result_dict)), f)


if __name__ == "__main__":
    main()
