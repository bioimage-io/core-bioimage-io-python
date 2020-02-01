from marshmallow import ValidationError


class PyBioException(Exception):
    pass

class PyBioMissingKwargException(PyBioException, TypeError):
    pass


class PyBioValidationException(PyBioException, ValidationError):
    pass


class InvalidDoiException(PyBioValidationException):
    pass
