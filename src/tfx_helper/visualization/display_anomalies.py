import os.path
from typing import Any

import tensorflow_data_validation as tfdv
import tensorflow_metadata as tfmd

from .utils import read_binary_proto


def display_anomalies(dir: str, split_name: str = "all") -> Any:
    """
    Display dataset anomalies in a notebook cell.

    Consumes ExampleValidator output.
    """
    path = os.path.join(dir, f"Split-{split_name}", "SchemaDiff.pb")
    anomalies = read_binary_proto(path, tfmd.proto.anomalies_pb2.Anomalies)
    return tfdv.display_anomalies(anomalies)
