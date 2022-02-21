import os.path
from typing import Any, Iterable, Mapping, Optional, Tuple

import tfx.v1 as tfx
from absl import logging
from ml_metadata.proto import metadata_store_pb2
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.types.channel import Channel

from .base import BasePipelineHelper
from .interface import DEFAULT_CUSTOM_CONFIG, Resources


class LocalPipelineHelper(BasePipelineHelper, arbitrary_types_allowed=True):
    model_push_destination: tfx.proto.PushDestination

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
        return tfx.components.Trainer(
            examples=examples,
            transform_graph=transform_graph,
            schema=schema,
            base_model=base_model,
            hyperparameters=hyperparameters,
            run_fn=run_fn,
            train_args=train_args,
            eval_args=eval_args,
            custom_config=dict(custom_config),
        )

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
        return tfx.components.Tuner(
            examples=examples,
            schema=schema,
            transform_graph=transform_graph,
            base_model=base_model,
            tuner_fn=tuner_fn,
            train_args=train_args,
            eval_args=eval_args,
            custom_config=dict(custom_config),
        )

    def construct_pusher(
        self,
        *,
        model: Optional[Channel] = None,
        model_blessing: Optional[Channel] = None,
        infra_blessing: Optional[Channel] = None,
        custom_config: Mapping[str, Any] = DEFAULT_CUSTOM_CONFIG,
    ) -> BaseComponent:
        return tfx.components.Pusher(
            model=model,
            model_blessing=model_blessing,
            infra_blessing=infra_blessing,
            push_destination=self.model_push_destination,
            custom_config=dict(custom_config),
        )

    def get_metadata_connection_config(self) -> metadata_store_pb2.ConnectionConfig:
        metadata_path = os.path.join(
            self.output_dir, "tfx_metadata", self.pipeline_name, "metadata.db"
        )
        logging.info("Pipeline will store metadata in %r", metadata_path)
        return tfx.orchestration.metadata.sqlite_metadata_connection_config(
            metadata_path
        )

    def create_and_run_pipeline(
        self,
        components: Iterable[BaseComponent],
        enable_cache: bool = False,
    ) -> None:
        logging.info(
            "Creating local pipeline name=%r, root=%r, enable_cache=%r",
            self.pipeline_name,
            self.pipeline_root,
            enable_cache,
        )
        metadata_connection_config = self.get_metadata_connection_config()
        pipeline = tfx.dsl.Pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root,
            components=list(components),
            enable_cache=enable_cache,
            metadata_connection_config=metadata_connection_config,
        )

        logging.info("Runnig pipeline using local DAG runner")
        tfx.orchestration.LocalDagRunner().run(pipeline)
        logging.info("Pipeline run finished")
