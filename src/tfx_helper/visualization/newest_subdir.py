import os.path
from dataclasses import dataclass
from typing import List

import tensorflow as tf
from absl import logging
from typing_extensions import Protocol


class NewestPathGetterInterface(Protocol):
    """Interface for tools for getting newest pipeline artifacts directory."""

    def __call__(self, component_name: str, output_name: str) -> str:
        """
        Get path to newest artifact.

        `component_name` is the name of the component,
        `output_name` is the name of component output.
        """


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


@dataclass
class NewestVertexAIPathGetter:
    """
    Tool for obtaining paths to newest artifact directories for your components.

    When running on VertexAI each run creates a new directory with epoch timestamp name.
    Within that directory a new directory with pipeline name prefix is created.
    Inside it there are directories for components, each prefixed with component name.
    Within the component directory there is one directory for each output.

    This tool helps in obtaining the most recent output directory
    for given component's output
    (that might not be what you want if you component failed
    or if you have multiple pipeline jobs running simultainously).
    """

    artifact_dir: str
    """
    Path to pipeline output directory on Google Cloud Storage.
    """
    pipeline_name: str
    """
    Name of the pipeline.
    """

    def __call__(self, component_name: str, output_name: str) -> str:
        pipeline_run_dirs = tf.io.gfile.glob(
            os.path.join(self.artifact_dir, self.pipeline_name, "*")
        )
        newest_run_dir = max(
            pipeline_run_dirs,
            key=lambda subdirectory: int(os.path.basename(subdirectory)),
        )
        logging.debug(
            "From %d subdirs selected %r as newest run dir",
            len(pipeline_run_dirs),
            newest_run_dir,
        )
        dirs = tf.io.gfile.glob(os.path.join(newest_run_dir, f"{self.pipeline_name}-*"))
        # The directories are timestamped pipelinename-20220721110503
        # the date should be string-sortable
        subdir = max(dirs, key=lambda dir: os.path.basename(dir))
        logging.debug("From %d dirs selected %r", len(dirs), subdir)
        component_dirs = tf.io.gfile.glob(os.path.join(subdir, f"{component_name}_*"))
        assert (
            len(component_dirs) == 1
        ), f"Expected a single component directory, but got {len(component_dirs)}"
        (component_dir,) = component_dirs
        result = os.path.join(component_dir, output_name)
        logging.debug("Selected %r as newest", result)
        return result
