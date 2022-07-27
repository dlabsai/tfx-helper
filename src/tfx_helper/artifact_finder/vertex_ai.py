import datetime
from dataclasses import dataclass, field
from typing import cast

from absl import logging

from .ml_metadata import MLMetadataQuery


@dataclass
class NewsetRunVertexAIPathGetter:
    """
    A tool for fetching artifacts from the newest pipeline run on VertexAI.

    Interfaces with VertexAI ML Metadata service.
    """

    pipeline_name: str
    """The name of the pipeline."""

    region: str
    """The GCP region."""

    project: str
    """The GCP project."""

    metadata_store_name: str = "default"
    """The name of the metadata store."""

    page_size: int = 20
    """The page size for listing items in ML Metadata."""

    query_util: MLMetadataQuery = field(init=False)
    """The initialized ML Metadata query util."""

    def __post_init__(self) -> None:
        self.query_util = MLMetadataQuery(
            pipeline_name=self.pipeline_name,
            region=self.region,
            project=self.project,
            metadata_store_name=self.metadata_store_name,
            page_size=self.page_size,
        )

    def __call__(self, component_name: str, output_name: str) -> str:
        runs = list(self.query_util.get_pipeline_runs())
        logging.debug("Found %d pipeline runs", len(runs))
        runs.sort(key=lambda run: cast(datetime.datetime, run.create_time))
        *_, most_recent_run = runs
        logging.debug(
            "Selected run %r created %s, updated %s",
            most_recent_run.name,
            most_recent_run.create_time,
            most_recent_run.update_time,
        )
        executions = self.query_util.get_executions(
            run_name=most_recent_run.name, component_name=component_name
        )
        (execution,) = executions
        logging.debug(
            "Found execution %r for component %r",
            execution.name,
            component_name,
        )
        # result = query.get_inputs_and_outputs(execution_name=execution.name)
        # query.get_output_artifacts(execution_name=execution.name)
        artifact_name = self.query_util.get_output_artifact_name(
            execution_name=execution.name, output_name=output_name
        )
        logging.debug(
            "Found artifact %r for output %r", artifact_name, output_name
        )
        artifact = self.query_util.get_artifact(artifact_name=artifact_name)
        artifact_uri: str = artifact.uri
        logging.debug("Returning artifact URI %r", artifact_uri)
        return artifact_uri
