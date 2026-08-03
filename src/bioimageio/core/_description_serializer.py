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
from bioimageio.spec.common import Sha256


class DescriptionSerializer:
    """Description serializer intended for client/server communication, NOT for sharing resource descriptions.

    This serializer only includes local files to keep the serialized package small.
    """

    STRING_ENCODING = "ascii"

    @staticmethod
    def serialize(rd: ResourceDescr) -> bytes:
        stream = save_bioimageio_package_to_stream(rd, local_files_only=True)
        _ = stream.seek(0)
        return stream.read()

    @classmethod
    def serialize_to_string(cls, rd: ResourceDescr) -> str:
        package_bytes = cls.serialize(rd)

        safe_bytes = cls._get_safe_bytes(package_bytes)
        serialized_str = safe_bytes.decode(cls.STRING_ENCODING)
        if len(serialized_str) <= 2083:
            raise RuntimeError(
                "Serialized model description should be longer than 2083 characters to not be treated as a URL on the server side."
            )
        return serialized_str

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
            raise ValueError(f"invalid serialized model package: {descr.get_reason()}")

        return descr

    @classmethod
    def serialize_to_string_and_hash(cls, rd: ResourceDescr) -> tuple[str, Sha256]:
        package_bytes = cls.serialize(rd)
        safe_bytes = cls._get_safe_bytes(package_bytes)
        serialized_str = safe_bytes.decode(cls.STRING_ENCODING)
        if len(serialized_str) <= 2083:
            raise RuntimeError(
                "Serialized model description should be longer than 2083 characters to not be treated as a URL on the server side."
            )
        sha256 = Sha256(hashlib.sha256(package_bytes).hexdigest())
        return serialized_str, sha256
