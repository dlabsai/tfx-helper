import os.path

import tensorflow as tf


def display_value_artifact(path: str) -> str:
    """Loads value of a value-type artifact as string."""
    assert tf.io.gfile.exists(path)
    if tf.io.gfile.isdir(path):
        # in local mode values are stored as `value` file within the directory
        file_path = os.path.join(path, "value")
    else:
        # on VertexAI the values are stored as file with artifact name
        file_path = path
    with tf.io.gfile.GFile(file_path, "r") as f:
        return f.read()
