import json
from dataclasses import dataclass, field
from typing import Iterable

from google.cloud import aiplatform_v1


# Helpers for creating filter definitions accoring to AIP-160
# (https://google.aip.dev/160)
def wrap_parens(value: str) -> str:
    return f"({value})"


def EQ(attr: str, value: str) -> str:
    return f"{attr} = {json.dumps(value)}"


def HAS(context: str, value: str) -> str:
    return f"{context}:{json.dumps(value)}"


def AND(*conditions: str) -> str:
    return wrap_parens(" AND ".join(conditions))


def OR(*conditions: str) -> str:
    return wrap_parens(" OR ".join(conditions))


def FUN(name: str, value: str) -> str:
    return f"{name}({json.dumps(value)})"


@dataclass
class MLMetadataQuery:
    """
    Tool for interfacing with VertexAI ML Metadata service.
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

    client: aiplatform_v1.MetadataServiceClient = field(init=False)
    """The configured instance of ML Metadata sevice client."""

    def __post_init__(self) -> None:
        api_endpoint = f"{self.region}-aiplatform.googleapis.com"
        self.client = aiplatform_v1.MetadataServiceClient(
            client_options={"api_endpoint": api_endpoint}
        )

    def _get_metadata_store_context(self) -> str:
        return (
            f"projects/{self.project}/"
            f"locations/{self.region}/"
            f"metadataStores/{self.metadata_store_name}"
        )

    def _get_pipeline_context(self) -> str:
        return (
            f"{self._get_metadata_store_context()}/"
            f"contexts/{self.pipeline_name}"
        )

    def get_pipeline_runs(
        self,
    ) -> Iterable[aiplatform_v1.types.context.Context]:
        """Search for pipeline runs for given pipeline."""
        request = aiplatform_v1.ListContextsRequest(
            parent=self._get_metadata_store_context(),
            filter=AND(
                EQ("schema_title", "system.PipelineRun"),
                HAS("parent_contexts", self._get_pipeline_context()),
            ),
            page_size=self.page_size,
        )
        results = self.client.list_contexts(request=request)
        return results

    def get_executions(
        self, *, run_name: str, component_name: str
    ) -> Iterable[aiplatform_v1.types.execution.Execution]:
        """Search for executions of given component within given pipeline run."""
        request = aiplatform_v1.ListExecutionsRequest(
            parent=self._get_metadata_store_context(),
            page_size=self.page_size,
            filter=AND(
                OR(
                    EQ("schema_title", "system.ContainerExecution"),
                    EQ("schema_title", "system.ResolverExecution"),
                ),
                EQ("display_name", component_name),
                FUN("in_context", run_name),
            ),
        )
        results = self.client.list_executions(request=request)
        return results

    def _get_inputs_and_outputs(
        self, *, execution_name: str
    ) -> aiplatform_v1.types.lineage_subgraph.LineageSubgraph:
        request = aiplatform_v1.QueryExecutionInputsAndOutputsRequest(
            execution=execution_name
        )
        results = self.client.query_execution_inputs_and_outputs(request=request)
        return results

    def get_output_artifact_name(self, *, execution_name: str, output_name: str) -> str:
        """Retrieve the name of a given output artifact from given execution."""
        inputs_and_outputs = self._get_inputs_and_outputs(execution_name=execution_name)
        output_events = filter(
            lambda event: event.type_ == aiplatform_v1.types.event.Event.Type.OUTPUT,
            inputs_and_outputs.events,
        )
        output_artifacts = filter(
            lambda event: event.labels.get("name") == output_name,
            output_events,
        )
        (event,) = output_artifacts
        artifact_name: str = event.artifact
        return artifact_name

    def get_artifact(
        self, *, artifact_name: str
    ) -> aiplatform_v1.types.artifact.Artifact:
        """Retrive given artifact's details."""
        request = aiplatform_v1.GetArtifactRequest(
            name=artifact_name,
        )
        result = self.client.get_artifact(request=request)
        return result
