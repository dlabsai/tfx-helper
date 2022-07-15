from typing import Any, Dict, List, Optional, Tuple

import tensorflow_model_analysis as tfma

# In Evaluator component usually you will want to compare the `candidate` model
# against a `baseline` model. You are more likely to want to view the stats
# of the `candidate` model, as it's the one you are training.
DEFAULT_MODEL_NAME = "candidate"


def passed_validation(dir: str) -> bool:
    """
    Check if model passed validation.

    Consumes `Evaluator` output.
    """
    validation_result = tfma.load_validation_result(dir)
    validation_passed: bool = validation_result.validation_ok
    return validation_passed


def get_slice_names(
    dir: str, model_name: str = DEFAULT_MODEL_NAME
) -> List[Tuple[Tuple[str, Any], ...]]:
    """
    Return slice names that are available for analysis.

    Consumes `Evaluator` output.
    """
    result = tfma.load_eval_result(
        output_path=dir, output_file_format=None, model_name=model_name
    )
    slice_names: List[Tuple[Tuple[str, Any], ...]] = result.get_slice_names()
    return slice_names


def display_metrics(
    dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    column: Optional[str] = None,
) -> Any:
    """
    Display metrics inside a notebook cell.

    Provide `column` if you configured `slicing_specs` in `EvalConfig`
    and want to see metrics breakdown by slice.

    Consumes `Evaluator` output.
    """
    result = tfma.load_eval_result(
        output_path=dir, model_name=model_name, output_file_format=None
    )
    return tfma.view.render_slicing_metrics(result, slicing_column=column)


def display_plots(
    dir: str,
    model_name: str = DEFAULT_MODEL_NAME,
    slice_key: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Display plots inside a notebook cell.

    Provide `slice_key` if you want to see plots for a particular slice
    of the dataset.

    Consumes `Evaluator` output.
    """
    result = tfma.load_eval_result(
        output_path=dir, model_name=model_name, output_file_format=None
    )
    if slice_key:
        slicing_spec = tfma.SlicingSpec(feature_values=slice_key)
    else:
        slicing_spec = None
    return tfma.view.render_plot(result, slicing_spec)
