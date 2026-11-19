import base64
import hashlib
from io import BytesIO
from zipfile import ZipFile

from bioimageio.spec import (
    InvalidDescr,
    ResourceDescr,
    load_description,
    save_bioimageio_package_to_stream,
)
from bioimageio.spec.common import FileDescr, RelativeFilePath, Sha256
from bioimageio.spec.model import v0_4


class DescriptionSerializer:
    """Description serializer intended for client/server communication, NOT for sharing resource descriptions.

    This serializer only includes local files and required v0.4 Python architectures
    to keep the serialized package small.
    """

    STRING_ENCODING = "ascii"

    @staticmethod
    def serialize(rd: ResourceDescr) -> bytes:
        stream = save_bioimageio_package_to_stream(rd, local_files_only=True)
        if isinstance(rd, v0_4.ModelDescr):
            weights = rd.weights.pytorch_state_dict
            if weights is not None and isinstance(
                weights.architecture, v0_4.CallableFromFile
            ):
                source = weights.architecture.source_file
                if isinstance(source, RelativeFilePath):
                    name = str(source)
                    if "\\" in name or ":" in name or ".." in source.path.parts:
                        raise ValueError(f"Unsafe architecture path: {name}")

                    contents = (
                        FileDescr(source=source, sha256=weights.architecture_sha256)
                        .get_reader()
                        .read()
                    )
                    with ZipFile(stream, "a") as package:
                        if name in package.namelist():
                            if package.read(name) != contents:
                                raise ValueError(f"Conflicting package file: {name}")
                        else:
                            package.writestr(name, contents)
        _ = stream.seek(0)
        return stream.read()

    @classmethod
    def serialize_to_string(cls, rd: ResourceDescr) -> str:
        package_bytes = cls.serialize(rd)

        safe_bytes = cls._get_safe_bytes(package_bytes)
        serialized_str = safe_bytes.decode(cls.STRING_ENCODING)
        serialized_str = cls._pad_to_min_length(serialized_str)
        return serialized_str

    MIN_LENGTH = 2084
    """Serialized strings of at most 2083 characters would be treated as a URL on the server side."""

    @classmethod
    def _pad_to_min_length(cls, serialized_str: str) -> str:
        # `base64.b64decode` discards non-alphabet characters, so newlines are safe padding
        return serialized_str + "\n" * (cls.MIN_LENGTH - len(serialized_str))

    @staticmethod
    def _get_safe_bytes(raw_bytes: bytes) -> bytes:
        return base64.b64encode(raw_bytes)

    @classmethod
    def deserialize_from_string(cls, serialized: str) -> ResourceDescr:
        package_bytes = base64.b64decode(serialized.encode(cls.STRING_ENCODING))
        return cls.deserialize(package_bytes)

    @staticmethod
    def deserialize(serialized: bytes) -> ResourceDescr:
        descr = load_description(ZipFile(BytesIO(serialized)), perform_io_checks=False)
        if isinstance(descr, InvalidDescr):
            raise TypeError(f"invalid serialized model package: {descr.get_reason()}")

        return descr

    @classmethod
    def serialize_to_string_and_hash(cls, rd: ResourceDescr) -> tuple[str, Sha256]:
        package_bytes = cls.serialize(rd)
        safe_bytes = cls._get_safe_bytes(package_bytes)
        serialized_str = safe_bytes.decode(cls.STRING_ENCODING)
        serialized_str = cls._pad_to_min_length(serialized_str)
        sha256 = Sha256(hashlib.sha256(package_bytes).hexdigest())
        return serialized_str, sha256
