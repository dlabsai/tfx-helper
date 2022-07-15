from typing import Any, Dict, Optional

import numpy as np
from absl import logging
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from .display_metrics import DEFAULT_MODEL_NAME
from .threshold_optimization import load_entries


def plot_binary_classification_confusion_matrix(
    dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    threshold: float = 0.5,
    slice_key: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Plots binary classification confusion matrix from model evaluation.

    Consumes Evaluator output.
    Uses `matplotlib`.
    """
    # Load evaluation step results
    entries = load_entries(dir=dir, model_name=model_name, slice_key=slice_key)
    # Select the entry with the most matching threshold
    entries.sort(key=lambda entry: abs(entry.threshold - threshold))
    data_at_threshold, *_rest = entries
    threshold_found = data_at_threshold.threshold
    logging.debug(
        "For sought threshold %f found data with threshold %f",
        threshold,
        threshold_found,
    )

    # Extract data points
    tp = data_at_threshold.tp
    fp = data_at_threshold.fp
    fn = data_at_threshold.fn
    tn = data_at_threshold.tn

    # Construct confustion matrix
    sum_actual_positives = tp + fn
    sum_actual_negatives = tn + fp
    count_matrix = np.array([[tp, fn], [fp, tn]])
    logging.debug("Confusion matrix counts %s", count_matrix)
    confusion_matrix = np.array(
        [
            np.array([tp, fn]) / sum_actual_positives,
            np.array([fp, tn]) / sum_actual_negatives,
        ]
    )
    logging.debug("Confusion matrix normalized %s", confusion_matrix)

    # Chart the confusion matrix
    plt.matshow(
        confusion_matrix, cmap="Blues", norm=Normalize(vmin=0.0, vmax=1.0)
    )
    ax = plt.gca()

    value: float
    for (x, y), value in np.ndenumerate(confusion_matrix):
        plt.text(
            y,
            x,
            f"{count_matrix[x, y]} ({value:.1%})",
            va="center",
            ha="center",
            bbox={"boxstyle": "round", "facecolor": "white", "edgecolor": "k"},
        )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.xticks([0, 1], ["positive", "negative"])
    plt.yticks([0, 1], ["positive", "negative"])
    ax.xaxis.set_label_position("top")
    plt.colorbar()
