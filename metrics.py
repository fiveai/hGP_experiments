from typing import Callable

import numpy as np
from emukit.core import ParameterSpace
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score

from models import Model
from test_functions import nan_mask_function


def create_f1_evaluator(
    space: ParameterSpace, performance: Callable[[np.ndarray], np.ndarray], fidelity: int = 10**5
) -> Callable[[Model], float]:
    x = space.sample_uniform(fidelity)

    nan_masked_performance = nan_mask_function(performance)
    actual_class = nan_masked_performance(x) < 0

    def f1_score_fn(model: Model) -> float:
        classification = model.classify(x)
        return f1_score(actual_class, classification)

    return f1_score_fn


def avg_precision(
    space: ParameterSpace, performance: Callable[[np.ndarray], np.ndarray], fidelity: int = 10**5
) -> Callable[[Model], float]:
    x = space.sample_uniform(fidelity)

    nan_masked_performance = nan_mask_function(performance)
    actual_class = nan_masked_performance(x) < 0

    def fn(model: Model) -> float:
        class_probability = model.class_probability(x)
        return average_precision_score(actual_class, class_probability)

    return fn
