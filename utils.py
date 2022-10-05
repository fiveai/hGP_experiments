from typing import Dict
from typing import List

import numpy as np


def summarise_results(
    result_dict: Dict[str, List[List[Dict[str, float]]]]
) -> Dict[str, List[Dict[str, Dict[str, np.ndarray]]]]:
    last_result = {name: [result[-1] for result in result_list] for name, result_list in result_dict.items()}
    metric_names = list(last_result[list(last_result.keys())[0]][0].keys())
    return {
        name: [
            {
                metric_name: {
                    "min": np.min([item[metric_name] for item in result_list], axis=0),
                    "max": np.max([item[metric_name] for item in result_list], axis=0),
                    "std": np.std([item[metric_name] for item in result_list], axis=0, ddof=1),
                    "mean": np.mean([item[metric_name] for item in result_list], axis=0),
                }
                for metric_name in metric_names
            }
        ]
        for name, result_list in last_result.items()
    }
