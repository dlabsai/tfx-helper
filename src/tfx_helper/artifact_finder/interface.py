from typing_extensions import Protocol


class ArtifactPathGetterInterface(Protocol):
    """Interface for tools for getting pipeline artifacts directories."""

    def __call__(self, component_name: str, output_name: str) -> str:
        """
        Get path to artifact.

        `component_name` is the name of the component,
        `output_name` is the name of component output.
        """
