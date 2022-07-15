import json
import os.path
from typing import Any, Dict

import tensorflow as tf


def display_hyperparams(
    dir: str,
) -> Dict[str, Any]:
    """
    Parse best found hyperparameters.

    Consumes `Tuner` output.
    """
    file_path = os.path.join(dir, "best_hyperparameters.txt")
    with tf.io.gfile.GFile(file_path, "r") as f:
        data = json.load(f)
    hparam_values: Dict[str, Any] = data[
        "values"
    ]  # skip the hparams search space dump
    return hparam_values
