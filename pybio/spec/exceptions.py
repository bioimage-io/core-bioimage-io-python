from marshmallow import ValidationError


class PyBioException(Exception):
    pass


class PyBioRunnerException(Exception):
    pass


class PyBioValidationException(PyBioException, ValidationError):
    pass


class PyBioUnconvertibleException(PyBioValidationException):
    pass
