import os.path
from dataclasses import dataclass
from typing import List

import tensorflow as tf
from absl import logging


def get_newest_local_subdirectory(directory: str) -> str:
    subdirectories: List[str] = tf.io.gfile.glob(os.path.join(directory, "*"))
    newest_subdir = max(
        subdirectories,
        key=lambda subdirectory: int(os.path.basename(subdirectory)),
    )
    logging.debug(
        "From %d subdirs selected %r as newest",
        len(subdirectories),
        newest_subdir,
    )
    return newest_subdir


@dataclass
class NewestLocalPathGetter:
    """
    Tool for obtaining paths to newest artifact directories for your components.

    When running locally each components creates a directory for artifacts with
    sequential integer name. This tool helps in obtaining the most recent one
    (that might not be what you want if you component failed).
    """

    artifact_dir: str
    """
    Path to pipeline output directory.
    """

    pipeline_name: str
    """
    Name of the pipeline.
    """

    def __call__(self, component_name: str, output_name: str) -> str:
        path = os.path.join(
            self.artifact_dir, self.pipeline_name, component_name, output_name
        )
        return get_newest_local_subdirectory(path)
