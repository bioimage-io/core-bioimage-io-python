import logging
import sys
from importlib import import_module

from spec import Source

logger = logging.getLogger(__name__)


def load_obj_from_source(source: Source):
    # model_spec = importlib.util.spec_from_file_location("module.name", file_path)
    # model_module = importlib.util.module_from_spec(model_spec)
    # model_spec.loader.exec_module(model_module)
    sys_path = list(sys.path)
    source_folder = source.path.parent.as_posix()
    sys.path.append(source_folder)
    logger.debug("temporarily adding %s to sys.path", source_folder)
    source_module_name = source.path.stem
    logger.debug("attempting to import %s", source_module_name)
    source_module = import_module(source_module_name)
    sys.path = sys_path
    return getattr(source_module, source.name)
