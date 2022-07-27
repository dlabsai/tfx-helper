import os.path
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import tensorflow as tf
import tensorflow_model_analysis as tfma
from absl import logging

from .display_value_artifact import display_value_artifact


@dataclass
class Entry:
    tp: int
    fp: int
    fn: int
    tn: int
    threshold: float


def load_entries(
    *,
    dir: str,
    model_name: str,
    slice_key: Optional[Dict[str, Any]] = None,
) -> List[Entry]:
    """
    Load confusion matrix statistics for multiple thresholds
    from `Evaluator` component output.
    """
    # Load evaluation step results
    result = tfma.load_eval_result(
        output_path=dir, output_file_format=None, model_name=model_name
    )
    if not result.model_location:
        # in case of pipeline running for the first time, there might be a single
        # unnamed model only in evaluation results. In such a case we will load
        # the data again without model name to retrieve the correct results.
        result = tfma.load_eval_result(output_path=dir, output_file_format=None)

    # Select data for given slice
    if slice_key is None:
        slice_spec = tfma.slicer.slicer_lib.SingleSliceSpec()
    else:
        slice_spec = tfma.SlicingSpec(feature_values=slice_key)
    data, config = tfma.view.util.get_plot_data_and_config(result.plots, slice_spec)
    # Extract confusion matrix data
    confusion_matrix_config = config["metricKeys"]["confusionMatrixPlot"]
    plot_data = data[confusion_matrix_config["metricName"]][
        confusion_matrix_config["dataSeries"]
    ]
    return [
        Entry(
            threshold=entry.get("threshold", 0.0),
            tp=int(entry.get("truePositives", 0.0)),
            fp=int(entry.get("falsePositives", 0.0)),
            fn=int(entry.get("falseNegatives", 0.0)),
            tn=int(entry.get("trueNegatives", 0.0)),
        )
        for entry in plot_data
    ]


def g_mean(entry: Entry) -> float:
    sensitivity = entry.tp / (entry.tp + entry.fn)
    specificity = entry.tn / (entry.fp + entry.tn)
    return (sensitivity * specificity) ** 2


def find_best_binary_classification_threshold(*, dir: str, model_name: str) -> float:
    entries = load_entries(dir=dir, model_name=model_name, slice_key=None)
    entries.sort(key=g_mean, reverse=True)
    best_entry, *_rest = entries
    logging.debug(
        "Best threshold found %s with G-mean %f",
        best_entry,
        g_mean(best_entry),
    )
    return best_entry.threshold


def load_best_threshold(path: str) -> float:
    return float(display_value_artifact(path))
