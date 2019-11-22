from marshmallow import ValidationError


class PyBioException(Exception):
    pass


class PyBioValidationException(PyBioException, ValidationError):
    pass


class InvalidDoiException(PyBioValidationException):
    pass
