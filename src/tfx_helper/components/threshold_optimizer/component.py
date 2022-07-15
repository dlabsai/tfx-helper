import tfx.v1 as tfx
from absl import logging
from tfx.dsl.component.experimental.decorators import component

from ...visualization.threshold_optimization import g_mean, load_entries


@component  # type:ignore
def BinaryClassificationThresholdOptimizer(
    model_evaluation: tfx.dsl.components.InputArtifact[  # type: ignore
        tfx.types.standard_artifacts.ModelEvaluation
    ],
    model_name: tfx.dsl.components.Parameter[str] = "candidate",  # type: ignore
) -> tfx.dsl.components.OutputDict(best_threshold=float):  # type: ignore
    """
    A component that tries to find the best binary classification threshold
    to use based on `Evaluator` component output.
    """
    entries = load_entries(
        dir=model_evaluation.uri, model_name=model_name, slice_key=None
    )
    entries.sort(key=g_mean, reverse=True)
    best_entry, *_rest = entries
    logging.info(
        "Best threshold found %s with G-mean %f",
        best_entry,
        g_mean(best_entry),
    )
    return {"best_threshold": best_entry.threshold}
