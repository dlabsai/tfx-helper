import os.path
from typing import Any, Callable, List

import tensorflow as tf
import tensorflow_data_validation as tfdv

get_basename: Callable[[str], str] = os.path.basename


def display_stats(
    dir: str, lhs_split: str = "Split-train", rhs_split: str = "Split-eval"
) -> Any:
    """
    Displays dataset statistics in a notebook cell.

    Two splits of the same dataset can be compared in visualization.
    Consumes `StatisticsGen` output.
    """
    directories: List[str] = tf.io.gfile.glob(os.path.join(dir, "Split-*"))
    names = map(get_basename, directories)
    splits = {
        name: os.path.join(directory, "FeatureStats.pb")
        for name, directory in zip(names, directories)
    }
    return tfdv.visualize_statistics(
        lhs_statistics=tfdv.load_stats_binary(splits[lhs_split]),
        lhs_name=lhs_split,
        rhs_statistics=tfdv.load_stats_binary(splits[rhs_split]),
        rhs_name=rhs_split,
    )


def compare_stats(
    *,
    left_dir: str,
    left_split: str = "train",
    left_label: str = "Train",
    right_dir: str,
    right_split: str = "all",
    right_label: str = "Inference",
) -> Any:
    """
    Displays dataset statistics in a notebook cell.

    Two different datasets can be compared in visualization.
    Consumes `StatisticsGen` output.
    """
    lhs_path = os.path.join(left_dir, f"Split-{left_split}", "FeatureStats.pb")
    lhs_stats = tfdv.load_stats_binary(lhs_path)
    rhs_path = os.path.join(
        right_dir, f"Split-{right_split}", "FeatureStats.pb"
    )
    rhs_stats = tfdv.load_stats_binary(rhs_path)
    return tfdv.visualize_statistics(
        lhs_statistics=lhs_stats,
        lhs_name=left_label,
        rhs_statistics=rhs_stats,
        rhs_name=right_label,
    )
