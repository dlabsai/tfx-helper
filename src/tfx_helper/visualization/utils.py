from typing import Type, TypeVar

from google.protobuf.message import Message

ProtoClass = TypeVar("ProtoClass", bound=Message)


def read_binary_proto(
    file_path: str, proto_class: Type[ProtoClass]
) -> ProtoClass:
    instance: ProtoClass = proto_class()
    with open(file_path, "rb") as f:
        data = f.read()
    instance.ParseFromString(data)
    return instance
