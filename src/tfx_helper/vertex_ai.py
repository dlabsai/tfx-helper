import json
import os.path
import tempfile
from types import MappingProxyType
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import tfx.v1 as tfx
from absl import logging
from google.cloud import aiplatform
from google.cloud.aiplatform import pipeline_jobs
from pydantic import Field
from tfx.dsl.components.base.base_component import BaseComponent
from tfx.types.channel import Channel

from .base import BasePipelineHelper
from .interface import DEFAULT_CUSTOM_CONFIG, Resources

EMPTY_OVERRIDES: Dict[str, Resources] = {}
DEFAULT_OVERRIDES = MappingProxyType(EMPTY_OVERRIDES)


class VertexAIPipelineHelper(BasePipelineHelper):
    google_cloud_project: str
    """
    The name of GCP project that this pipeline should run in.
    """

    google_cloud_region: str
    """
    The region to run the pipeline in.
    """

    docker_image: str
    """
    The name (location) of the tfx-descended custom docker image built for the project.
    """

    service_account: str
    """
    The name of the service account to use when running the pipeline.
    """

    trainer_machine_type: str = "n1-standard-4"
    """
    The machine type to use for running a training job.
    """

    trainer_accelerator_type: Optional[str] = None
    """
    The (optional) GPU to request for running a training job.
    For example "NVIDIA_TESLA_K80".
    """

    serving_machine_type: str = "n1-standard-4"
    """
    The machine type to use for endpoint serving.
    """

    serving_accelerator_type: Optional[str] = None
    """
    The (optional) GPU to request for serving deployed endpoint.
    For example "NVIDIA_TESLA_K80"
    """

    serving_endpoint_name: str
    """
    The name of the endpoint to serve predictions at.
    """

    num_parallel_trials: int = 1
    """
    Number of hyperparameter tuning trails to execute in parallel.

    The number will indicate how many machines are going to be used at the same time
    for running HP tuning workers.
    """

    use_dataflow: bool = False
    """
    Whether to use Dataflow for Beam-powered components.
    """

    resource_overrides: Mapping[str, Resources] = Field(
        default_factory=lambda: DEFAULT_OVERRIDES
    )
    """
    Definition of resources needs of particular components.

    The key is the string ID of a component - which is usually the name of the class
    unless explicitly overridden using `with_id('<new id>')`.

    By default all components are running on `e2-standard-4` machine.
    This setting allows requesting bigger machine sizes if a component needs it.
    The machine type will be automatically selected by Vertex AI to provide at least
    the resources requested here.
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
        machine_spec: Dict[str, Any] = {
            "machine_spec": {
                "machine_type": self.trainer_machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": self.docker_image,
            },
        }
        if self.trainer_accelerator_type is not None:
            machine_spec["machine_spec"][
                "accelerator_type"
            ] = self.trainer_accelerator_type
            machine_spec["machine_spec"]["accelerator_count"] = 1
        vertex_job_spec = {
            "project": self.google_cloud_project,
            "worker_pool_specs": [machine_spec],
        }

        return tfx.extensions.google_cloud_ai_platform.Trainer(
            examples=examples,
            transform_graph=transform_graph,
            schema=schema,
            base_model=base_model,
            hyperparameters=hyperparameters,
            run_fn=run_fn,
            train_args=train_args,
            eval_args=eval_args,
            custom_config={
                **custom_config,
                tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
                tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: self.google_cloud_region,
                tfx.extensions.google_cloud_ai_platform.TRAINING_ARGS_KEY: vertex_job_spec,
            },
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
        tune_args: Optional[tfx.proto.TuneArgs] = None,
        custom_config: Mapping[str, Any] = DEFAULT_CUSTOM_CONFIG,
    ) -> BaseComponent:
        machine_spec: Dict[str, Any] = {
            "machine_spec": {
                "machine_type": self.trainer_machine_type,
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": self.docker_image,
            },
        }
        if self.trainer_accelerator_type is not None:
            machine_spec["machine_spec"][
                "accelerator_type"
            ] = self.trainer_accelerator_type
            machine_spec["machine_spec"]["accelerator_count"] = 1
        vertex_job_spec = {
            "worker_pool_specs": [
                machine_spec  # replicas will be configured just like master
            ],
        }

        training_inputs = {
            "job_spec": vertex_job_spec,
            "project": self.google_cloud_project,
        }

        return tfx.extensions.google_cloud_ai_platform.Tuner(
            examples=examples,
            transform_graph=transform_graph,
            schema=schema,
            base_model=base_model,
            tuner_fn=tuner_fn,
            train_args=train_args,
            eval_args=eval_args,
            tune_args=tfx.proto.TuneArgs(num_parallel_trials=self.num_parallel_trials),
            custom_config={
                **custom_config,
                tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
                tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: self.google_cloud_region,
                tfx.extensions.google_cloud_ai_platform.experimental.TUNING_ARGS_KEY: training_inputs,
                tfx.extensions.google_cloud_ai_platform.experimental.REMOTE_TRIALS_WORKING_DIR_KEY: os.path.join(
                    self.pipeline_root, "trials"
                ),
            },
        )

    def construct_pusher(
        self,
        *,
        model: Optional[Channel] = None,
        model_blessing: Optional[Channel] = None,
        infra_blessing: Optional[Channel] = None,
        push_destination: Optional[tfx.proto.PushDestination] = None,
        custom_config: Mapping[str, Any] = DEFAULT_CUSTOM_CONFIG,
    ) -> BaseComponent:
        vertex_serving_spec: Dict[str, Any] = {
            "project_id": self.google_cloud_project,
            "endpoint_name": self.serving_endpoint_name,
            "machine_type": self.serving_machine_type,
        }

        if self.serving_accelerator_type is None:
            serving_image = (
                "europe-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-7:latest"
            )
        else:
            serving_image = (
                "europe-docker.pkg.dev/vertex-ai/prediction/tf2-gpu.2-7:latest"
            )
            vertex_serving_spec["accelerator_type"] = self.serving_accelerator_type
            vertex_serving_spec["accelerator_count"] = 1

        return tfx.extensions.google_cloud_ai_platform.Pusher(
            model=model,
            model_blessing=model_blessing,
            infra_blessing=infra_blessing,
            custom_config={
                **custom_config,
                tfx.extensions.google_cloud_ai_platform.ENABLE_VERTEX_KEY: True,
                tfx.extensions.google_cloud_ai_platform.VERTEX_REGION_KEY: self.google_cloud_region,
                tfx.extensions.google_cloud_ai_platform.VERTEX_CONTAINER_IMAGE_URI_KEY: serving_image,
                tfx.extensions.google_cloud_ai_platform.SERVING_ARGS_KEY: vertex_serving_spec,
            },
        )

    def create_and_run_pipeline(
        self,
        components: Iterable[BaseComponent],
        enable_cache: bool = False,
    ) -> None:
        beam_pipeline_args: Optional[List[str]] = None
        if self.use_dataflow:
            logging.info("Including Dataflow Beam configuration")
            beam_pipeline_args = [
                "--runner=DataflowRunner",
                f"--project={self.google_cloud_project}",
                f"--worker_harness_container_image={self.docker_image}",
                "--experiments=use_runner_v2",
            ]

        pipeline = tfx.dsl.Pipeline(
            pipeline_name=self.pipeline_name,
            pipeline_root=self.pipeline_root,
            components=list(components),
            enable_cache=enable_cache,
            beam_pipeline_args=beam_pipeline_args,
        )

        with tempfile.NamedTemporaryFile(
            "w+", suffix=".json", prefix="pipeline."
        ) as temp_pipeline_file:
            logging.info(
                "Pipeline definition will be stored in temporary file %r",
                temp_pipeline_file.name,
            )
            runner = tfx.orchestration.experimental.KubeflowV2DagRunner(
                config=tfx.orchestration.experimental.KubeflowV2DagRunnerConfig(
                    default_image=self.docker_image
                ),
                output_filename=temp_pipeline_file.name,
            )
            logging.info("Compile VertexAI pipeline definition")
            runner.run(pipeline)  # only writes pipeline definition to file

            # Read the definition and update it with resouce requests
            # (not supported by TFX, but supported by the underlying VertexAI platform)
            logging.info("Patching pipeline definition")
            pipeline_definition = json.load(temp_pipeline_file)
            executors_definition: Dict[str, Dict[str, Any]] = pipeline_definition[
                "pipelineSpec"
            ]["deploymentSpec"]["executors"]
            for component_id, resource in self.resource_overrides.items():
                try:
                    component_definition = executors_definition[
                        f"{component_id}_executor"
                    ]
                except KeyError:
                    logging.warning(
                        "Could not find component %r for resource override %r",
                        component_id,
                        resource,
                    )
                else:
                    logging.info(
                        "Component %r will run with resource overrides %r",
                        component_id,
                        resource,
                    )
                    component_definition["container"]["resources"] = {
                        "cpuLimit": resource.cpu,
                        "memoryLimit": resource.memory,
                    }
            temp_pipeline_file.seek(0)
            temp_pipeline_file.truncate(0)
            json.dump(pipeline_definition, temp_pipeline_file)
            temp_pipeline_file.flush()

            logging.info("Submiting VertexAI pipeline job")
            aiplatform.init(
                project=self.google_cloud_project,
                location=self.google_cloud_region,
            )
            job = pipeline_jobs.PipelineJob(
                template_path=temp_pipeline_file.name, display_name=self.pipeline_name
            )
            job.submit(
                service_account=self.service_account,
            )
            logging.info("Job submitted")
