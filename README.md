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

In addition we provide a set of helper functions for visualization of TFX pipeline artifacts.

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

### Viewing pipeline artifacts

The TFX pipeline artifacts can be visualized inside Jupyter Notebook cells using
(mostly) custom interactive widgets. In order to be able to display those widgets,
the following prerequisites need to be fulfilled:

* widget extensions need to be installed,

    ```sh
    jupyter nbextension enable --py widgetsnbextension --sys-prefix
    jupyter nbextension install --py --symlink tensorflow_model_analysis --sys-prefix
    jupyter nbextension enable --py tensorflow_model_analysis --sys-prefix
    ```

* notebook needs to be trusted.

#### Getting recent artifact directory

Each TFX artifact is stored in the TFX pipeline's output location.
Depending on whether you run your pipeline locally or on VertexAI the structure
of the directories is different. We provide helpers that can assist you in retrieving
the newest artifact locations. In local environment the helper uses directory traversal
to find the correct path. With VertexAI the helper communicates with VertexAI ML Metadata
Service.

For local pipeline runs:

```python
from tfx_helper.artifact_finder.local import NewestLocalPathGetter
from tfx_helper.artifact_finder.interface import ArtifactPathGetterInterface

path_getter: ArtifactPathGetterInterface = NewestLocalPathGetter(
    artifact_dir=pipeline_output, pipeline_name=pipeline_name
)

best_hparams_path = path_getter('Tuner', 'best_hyperparameters')
evaluation_path = path_getter('Evaluator', 'evaluation')
```

For VertexAI pipeline runs:

```python
from tfx_helper.artifact_finder.interface import ArtifactPathGetterInterface
from tfx_helper.artifact_finder.vertex_ai import NewsetRunVertexAIPathGetter

path_getter: ArtifactPathGetterInterface = NewsetRunVertexAIPathGetter(
    pipeline_name=pipeline_name, region=gcp_region, project=gcp_project
)

best_hparams_path = path_getter('Tuner', 'best_hyperparameters')
evaluation_path = path_getter('Evaluator', 'evaluation')
```

The obtained directory can be used with our set of helper functions that help you
preview the artifacts in an easy manner.

Please note that the name of a pipeline component can be customized. If you define
in your pipeline:

```python
stats_gen = tfx.components.StatisticsGen(...)
```

then you should use:

```python
evaluation_path = path_getter('StatisticsGen', 'statistics')
```

but if you customize the name:

```
stats_gen = tfx.components.StatisticsGen(...).with_id("raw_stats_gen")
```

then you should use:

```python
evaluation_path = path_getter('raw_stats_gen', 'statistics')
```

#### Viewing dataset statistics

Use the `display_stats` helper to compare statistics of two splits of the same dataset:


```python
from tfx_helper.visualization.display_stats import display_stats

display_stats(path_getter('raw_stats_gen', 'statistics'))
```

Use the `compare_stats` helper to compare statistics between two datasets.

```python
from tfx_helper.visualization.display_stats import compare_stats

compare_stats(
    left_dir=path_getter('raw_stats_gen', 'statistics'),
    right_dir=path_getter('inference_stats_gen', 'statistics')
)
```

#### Viewing dataset schema

Use the `display_schema` helper to preview dataset schema:

```python
from tfx_helper.visualization.display_schema import display_schema

display_schema(path_getter('SchemaGen', 'schema'))
```

#### Viewing dataset anomalies

If you use `ExampleValidator` component, you can preview its anomalies detection report
using `display_anomalies` helper:

```python
from tfx_helper.visualization.display_anomalies import display_anomalies

display_anomalies(path_getter('ExampleValidator', 'anomalies'), split_name='all')
```

#### Viewing best hyper parameters

To view the values of best hyperparameters chosen by the `Tuner` component, you can use
the `display_hyperparams` helper:

```python
from tfx_helper.visualization.display_hyperparams import display_hyperparams

display_hyperparams(path_getter('Tuner', 'best_hyperparameters'))
```

#### Viewing model evaluation results

If you use `slicing_specs` in `EvalConfig` to your `Evaluator` component, then you might
be willing to view what kind of slices were produced during evaluation:

```python
from tfx_helper.visualization.display_metrics import get_slice_names

get_slice_names(path_getter('Evaluator', 'evaluation'))
```

To view the overall metrics for your model use `display_metrics`:

```python
from tfx_helper.visualization.display_metrics import display_metrics

display_metrics(path_getter('Evaluator', 'evaluation'))
```

To view metrics by slice use:

```python
display_metrics(path_getter('Evaluator', 'evaluation'), column='HomePlanet')
```

To view the overall plots for your model use `display_plots`:

```python
from tfx_helper.visualization.display_metrics import display_plots

display_plots(path_getter('Evaluator', 'evaluation'))
```

To view plots for a specific slice use:

```python
display_plots(path_getter('Evaluator', 'evaluation'), slice_key={'HomePlanet': 'Earth'})
```

To check whether the candidate model passed validation (when you have thresholds configured in `EvalConfig`)
use `passed_validation`:

```python

from tfx_helper.visualization.display_metrics import passed_validation

passed_validation(path_getter('Evaluator', 'evaluation'))
```

#### Viewing binary classification confusion matrices

To view a binary classification confusion matrix constructed from the evaluation result
of your model (data availability determined by presence of `ConfusionMatrixPlot`
in `EvalConfig` to your `Evaluator` component):

```python
from tfx_helper.visualization.confusion_matrix import plot_binary_classification_confusion_matrix

plot_binary_classification_confusion_matrix(path_getter('Evaluator', 'evaluation'))
```

We provide a pipeline component for finding the best (by geometric mean) threshold
to use from the thresholds gathered during evaluation:

```python
from tfx_helper.components.threshold_optimizer.component import (
    BinaryClassificationThresholdOptimizer,
)

threshold_optimizer = BinaryClassificationThresholdOptimizer(
    model_evaluation=evaluator.outputs["evaluation"]
)
```

To view the confusion plot with the optimized threshold use:

```python
from tfx_helper.visualization.threshold_optimization import load_best_threshold

best_threshold = load_best_threshold(
    path_getter('ThresholdOptimizer', 'best_threshold')
)
plot_binary_classification_confusion_matrix(
    path_getter('Evaluator', 'evaluation'), threshold=best_threshold
)
```

## More info

Link to article describing creation of *TFX* pipeline for sentiment analysis using
this helper library: [https://dlabs.ai/resources/courses/bert-sentiment-analysis-on-vertex-ai-using-tfx/](https://dlabs.ai/resources/courses/bert-sentiment-analysis-on-vertex-ai-using-tfx/)
