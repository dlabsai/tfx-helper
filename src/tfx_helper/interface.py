from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Dict, Iterable, Mapping, Optional

import tfx.v1 as tfx
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.types.channel import Channel
from typing_extensions import Protocol

EMPTY_CONFIG: Dict[str, Any] = {}
DEFAULT_CUSTOM_CONFIG = MappingProxyType(EMPTY_CONFIG)


@dataclass
class Resources:
    """
    Pipeline step resource requirements definition.
    """

    cpu: int
    """Number of CPUs."""

    memory: int
    """Number of RAM memory GB."""


class PipelineHelperInterface(Protocol):
    """
    Describes a tool that hides the complexities of creating TFX components that
    can be run on both local setup and in the cloud.
    """

    def construct_trainer(
        self,
        *,
        examples: Optional[Channel] = None,
        transform_graph: Optional[Channel] = None,
        schema: Optional[Channel] = None,
        base_model: Optional[Channel] = None,
        hyperparameters: Optional[Channel] = None,
        run_fn: str,
        train_args: Optional[tfx.proto.TrainArgs] = None,
        eval_args: Optional[tfx.proto.EvalArgs] = None,
        custom_config: Mapping[str, Any] = DEFAULT_CUSTOM_CONFIG,
    ) -> BaseComponent:
        """Create an appropriate trainer component from given arguments."""

    def construct_tuner(
        self,
        *,
        examples: Channel,
        schema: Optional[Channel] = None,
        transform_graph: Optional[Channel] = None,
        base_model: Optional[Channel] = None,
        tuner_fn: str,
        train_args: Optional[tfx.proto.TrainArgs] = None,
        eval_args: Optional[tfx.proto.EvalArgs] = None,
        custom_config: Mapping[str, Any] = DEFAULT_CUSTOM_CONFIG,
    ) -> BaseComponent:
        """Create an appropriate tuner component from given arguments."""

    def construct_pusher(
        self,
        *,
        model: Optional[Channel] = None,
        model_blessing: Optional[Channel] = None,
        infra_blessing: Optional[Channel] = None,
        custom_config: Mapping[str, Any] = DEFAULT_CUSTOM_CONFIG,
    ) -> BaseComponent:
        """Create an appropriate pusher component from given arguments."""

    def create_and_run_pipeline(
        self,
        components: Iterable[BaseComponent],
        enable_cache: bool = False,
    ) -> None:
        """Create and run a pipeline from given set of components and resources."""

    @property
    def pipeline_root(self) -> str:
        """Get the pipeline artifacts root directory."""
