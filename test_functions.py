from typing import Callable

import numpy as np


def performance_function(x: np.ndarray) -> np.ndarray:
    out = np.cos(8 * x)
    is_nan = (0.215 < x) & (x < 0.6)
    out[is_nan] = np.nan
    return out


def nan_mask_function(
    function: Callable[[np.ndarray], np.ndarray], mask_to: float = 1
) -> Callable[[np.ndarray], np.ndarray]:
    def masked_fn(x: np.ndarray) -> np.ndarray:
        result = function(x)
        is_nan = np.isnan(result)
        result[is_nan] = mask_to
        return result

    return masked_fn


def pem(x: np.ndarray) -> np.ndarray:
    detected = np.abs(x) < 60
    return detected


def car_performance(x_0: np.ndarray, vx: np.ndarray) -> np.ndarray:
    detected = pem(x_0)
    rescale = 20

    safe_threshold = 20  # set due to stopping distance
    ego_x_0 = 0
    ego_a = 2
    t = np.linspace(0, 20, 1000)
    ego_x = ego_x_0 + ego_a * t**2 / 2
    adversary_x = np.moveaxis(np.array([x_0 + vx * t_i for t_i in t]), 0, -1)
    min_distance = np.min(np.abs(ego_x - adversary_x), axis=-1)
    g = (min_distance - safe_threshold) / rescale
    g[np.logical_and(detected, g < 0)] = np.nan
    return g


def car_performance_2(x_0: np.ndarray, vx: np.ndarray) -> np.ndarray:
    rescale = 20
    safe_threshold = 20  # set due to stopping distance
    # ego_x_0 = 0
    ego_a = 2
    min_distance = np.maximum(-(x_0 + vx**2 / (2 * ego_a)), np.zeros_like(vx))
    join_road = np.logical_or(min_distance > safe_threshold, ~pem(x_0))
    g = np.where(join_road, min_distance - safe_threshold, np.nan)
    return g / rescale
