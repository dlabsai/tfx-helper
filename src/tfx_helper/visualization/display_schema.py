import os.path
from typing import Any

import tensorflow_data_validation as tfdv


def display_schema(dir: str) -> Any:
    """
    Display dataset schema inside a notebook cell.

    Consumes SchemaGen output.
    """
    schema = tfdv.load_schema_text(os.path.join(dir, "schema.pbtxt"))
    return tfdv.display_schema(schema)
