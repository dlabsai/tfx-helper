from typing import Type, TypeVar

import tensorflow as tf
from google.protobuf.message import Message

ProtoClass = TypeVar("ProtoClass", bound=Message)


def read_binary_proto(file_path: str, proto_class: Type[ProtoClass]) -> ProtoClass:
    instance: ProtoClass = proto_class()
    with tf.io.gfile.GFile(file_path, "rb") as f:
        data = f.read()
    instance.ParseFromString(data)
    return instance
