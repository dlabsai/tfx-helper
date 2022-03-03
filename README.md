# tfx-helper

A helper library for TFX

## Why?

This package contains small utilities that help in creation of [TFX](https://github.com/tensorflow/tfx) pipelines:

* supports running pipeline locally or on *Vertex AI Pipelines* (with all training, tuning and serving also happening inside *Vertex AI*),
* abstracts local vs cloud execution environment outside of the pipeline definition (no need for `if use_gcp:` conditions inside your pipeline code - write a uniform pipeline creation code and run it on both local and cloud),
* construct complex `custom_config` for *TFX* extension components for you (configuration of extension components is complex and not well documented - we did the research for you and are exposing a simple API),
* enable GPU in training/tuning/serving with a single argument,
* enable declaring per-component resource requirements (you can now run `Evaluator` component on a beefier machine if you have a large model),
* use generator function syntax in pipeline definition to avoid boilerplate,
* avoid passing a hundred parameters into your pipeline definition (cloud configuration, like `service_account` is now only part of cloud-targeted runner)

## How?

### Install

```sh
pip install tfx-helper
```

### Pipeline definition

1. Use our helper component interface in you pipeline definition.
1. Return a collection of components.
1. For multi-version components (`Trainer`, `Tuner`, `Pusher`) construction use the helper.

```python
from tfx_helper.interface import PipelineHelperInterface

def create_pipeline(
    pipeline_helper: PipelineHelperInterface, # pass in the helper as interface
    *,
    # all your pipeline parameters
    train_epochs: int,  # maximum number of training epochs in trainer
    ... # other parameters
) -> Iterable[BaseComponent]: # return a collection of components
    ...
    # create `Transform` in the usual way
    transform = tfx.components.Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        preprocessing_fn="models.preprocessing.preprocessing_fn",
        splits_config=tfx.proto.SplitsConfig(
            analyze=["train", "valid"],
            transform=["train", "valid", "eval"],
        ),
    )
    yield transform
    ...
    # use the helper to create a `Trainer` in a uniform way
    trainer = pipeline_helper.construct_trainer(
        run_fn="models.model.run_fn",
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        hyperparameters=hparams,
        train_args=tfx.proto.TrainArgs(splits=["train"]),
        eval_args=tfx.proto.EvalArgs(splits=["valid"]),
        # custom parameters to the training callback
        custom_config={"epochs": train_epochs, "patience": train_patience},
    )
    yield trainer
    ...
```

### Pipeline local runner

Create a pipeline runner that will take your uniform pipeline definition and materialize
it for running locally (through `DirectRunner`):

```python
from tfx_helper.local import LocalPipelineHelper


def run() -> None:
    """Create and run a pipeline locally."""
    input_dir = ...
    output_dir = ...
    serving_model_dir = ...
    # Create pipeline helper instance of local flavour.
    pipeline_helper = LocalPipelineHelper(
        pipeline_name="sentimentanalysis",
        output_dir=output_dir,
        # Where should the model be pushed to
        model_push_destination=tfx.proto.PushDestination(
            filesystem=tfx.proto.PushDestination.Filesystem(
                base_directory=serving_model_dir
            )
        ),
    )

    components = create_pipeline(
        # Pass our pipeline helper instance
        pipeline_helper,
        # The rest of the parameters are pipeline-specific.
        data_path=input_dir,
        ... # other arguments
    )
    # Run the pipeline
    pipeline_helper.create_and_run_pipeline(components)
```

Notice that no cloud-specific configuration was needed neither in the runner nor in the
pipeline definition.

### Pipeline cloud runner

Create a pipline runner that will take you uniform pipeline definition and materialize
it for running in the cloud (on *Vertex AI Pipelines* through `KubeflowV2DagRunner`):

```python
from tfx_helper.interface import Resources
from tfx_helper.vertex_ai import VertexAIPipelineHelper

def run() -> None:
    output_dir = "gs://..."
    # minimal (less than the standard `e2-standard-4`) resource for components
    # that won't execute computations
    minimal_resources = Resources(cpu=1, memory=4)
    # create a helper instance of cloud flavour
    pipeline_helper = VertexAIPipelineHelper(
        pipeline_name="...",
        output_dir=output_dir,
        google_cloud_project="...",
        google_cloud_region="europe-west4",
        # all the components will use our custom image for running
        docker_image="europe-west4-docker.pkg.dev/.../...-repo/...-image:latest",
        service_account="...@....iam.gserviceaccount.com",
        # name of the Vertex AI Endpoint
        serving_endpoint_name="...",
        # Number of parallel hyperparameter tuning trails
        num_parallel_trials=2,
        # GPU for Trainer and Tuner components
        trainer_accelerator_type="NVIDIA_TESLA_T4",
        # Machine type for Trainer and Tuner components
        trainer_machine_type="n1-standard-4",
        # GPU for serving endpoint
        serving_accelerator_type="NVIDIA_TESLA_T4",
        # Machine type for serving endpoint
        serving_machine_type="n1-standard-4",
        # Override resource requirements of components. The dictionary key is the ID
        # of the component (usually class name, unless changed with `with_id` method).
        resource_overrides={
            # evaluator needs more RAM than standard machine can provide
            "Evaluator": Resources(cpu=16, memory=32),
            # training is done as Vertex job on a separate machine
            "Trainer": minimal_resources,
            # tuning is done as Vertex job on a separate set of machines
            "Tuner": minimal_resources,
            # pusher is just submitting a job
            "Pusher": minimal_resources,
        },
    )
    # Run the pipeline
    components = create_pipeline(
        pipeline_helper,
        # Input data in Cloud Storage
        data_path="gs://...",
        ... # other arguments
    )
    # Run the pipeline
    pipeline_helper.create_and_run_pipeline(components, enable_cache=True)
```

## More info

Link to article describing creation of *TFX* pipeline for sentiment analysis using
this helper library: [LINK_GOES_HERE](https://dlabs.ai/blog/)
