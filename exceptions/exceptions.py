class BaseSimCLRException(Exception):
    """Base exception"""


class InvalidBackboneError(BaseSimCLRException):
    """Raised when the choice of backbone Convnet is invalid."""


class InvalidDatasetSelection(BaseSimCLRException):
    """Raised when the choice of dataset is invalid."""


class InvalidTrainingMode(BaseSimCLRException):
    """Raised when the choice of training mode is invalid"""


class InvalidEstimationMode(BaseSimCLRException):
    """Raised when the choice of estimation mode is invalid"""
