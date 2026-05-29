import base64
from io import BytesIO
from zipfile import ZipFile

from loguru import logger

from bioimageio.spec import (
    InvalidDescr,
    ResourceDescr,
    get_package_content,
    load_description,
    save_bioimageio_package_to_stream,
)


class DescriptionSerializer:
    """Description serializer intended for client/server communication, NOT for sharing resource descriptions.

    This serializer only includes local files to keep the serialized package small.
    """

    @staticmethod
    def serialize(rd: ResourceDescr) -> bytes:
        package_content = get_package_content(rd, local_files_only=True)
        local_files = {
            k: v for k, v in package_content.items() if not isinstance(v, dict)
        }
        if local_files:
            logger.warning(
                f"Model description references {len(local_files)} local files that"
                + " will be sent to the server with every prediction request."
            )

        stream = save_bioimageio_package_to_stream(rd, local_files_only=True)
        _ = stream.seek(0)
        return stream.read()

    @classmethod
    def serialize_to_string(cls, rd: ResourceDescr) -> str:
        package_bytes = cls.serialize(rd)

        # Encode binary package bytes as ASCII text so it can be JSON-serialized.
        return base64.b64encode(package_bytes).decode("ascii")

    @classmethod
    def deserialize_from_string(cls, serialized: str) -> ResourceDescr:
        package_bytes = base64.b64decode(serialized.encode("ascii"))
        return cls.deserialize(package_bytes)

    @staticmethod
    def deserialize(serialized: bytes) -> ResourceDescr:
        descr = load_description(ZipFile(BytesIO(serialized)), perform_io_checks=False)
        if isinstance(descr, InvalidDescr):
            raise ValueError(f"invalid serialized model package: {descr.get_reason()}")

        return descr
