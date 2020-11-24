from marshmallow import ValidationError


class PyBioException(Exception):
    pass


class PyBioMissingKwargException(PyBioException, TypeError):
    pass


class PyBioValidationException(PyBioException, ValidationError):
    pass

class PyBioUnconvertableException(PyBioValidationException):
    pass


class InvalidDoiException(PyBioValidationException):
    pass
