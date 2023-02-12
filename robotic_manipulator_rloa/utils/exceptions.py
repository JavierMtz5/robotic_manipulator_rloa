from __future__ import annotations

from typing import Optional


class FrameworkException(Exception):

    def __init__(self, message: str) -> None:
        """
        Creates new FrameworkException. Base class that must be extended by any custom exception.
        Args:
            message: Info about the exception.
        """
        Exception.__init__(self, message)
        self.message = message

    def __str__(self) -> str:
        """ Returns the string representation of the object """
        return self.__class__.__name__ + ': ' + self.message

    def set_message(self, value: str) -> FrameworkException:
        """
        Set the message to be printed on terminal.
        Args:
            value: Message with info about exception.
        Returns:
            FrameworkException
        """
        self.message = value
        return self


class InvalidManipulatorFile(FrameworkException):
    """
    Exception raised when the URDF/SDF file received cannot be loaded with
    Pybullet's loadURDF/loadSDF methods.
    """
    message = 'The URDF/SDF file received is not valid'

    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        FrameworkException.__init__(self, self.message)


class InvalidHyperParameter(FrameworkException):
    """
    Exception raised when the user tries to set an invalid value on a hyper-parameter
    """
    message = 'The hyperparameter received is not valid'

    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        FrameworkException.__init__(self, self.message)


class InvalidEnvironmentParameter(FrameworkException):
    """
    Exception raised when the Environment is initialized with invalid parameter/parameters.
    """
    message = 'The Environment parameter received is not valid'

    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        FrameworkException.__init__(self, self.message)


class InvalidNAFAgentParameter(FrameworkException):
    """
    Exception raised when the NAFAgent is initialized with invalid parameter/parameters.
    """
    message = 'The NAF Agent parameter received is not valid'

    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        FrameworkException.__init__(self, self.message)


class EnvironmentNotInitialized(FrameworkException):
    """
    Exception raised when the Environment has not yet been initialized and the user tries to
    call a method which requires the Environment to be initialized.
    """
    message = 'The Environment is not yet initialized. The environment can be initialized via the ' \
              'initialize_environment() method'

    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        FrameworkException.__init__(self, self.message)


class NAFAgentNotInitialized(FrameworkException):
    """
    Exception raised when the NAFAgent has not yet been initialized and the user tries to
    call a method which requires the NAFAgent to be initialized.
    """
    message = 'The NAF Agent is not yet initialized. The agent can be initialized via the ' \
              'initialize_naf_agent() method'

    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        FrameworkException.__init__(self, self.message)


class MissingWeightsFile(FrameworkException):
    """
    Exception raised when the user loads pretrained weights from an invalid location.
    """
    message = 'The weight file provided does not exist'

    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        FrameworkException.__init__(self, self.message)


class ConfigurationIncomplete(FrameworkException):
    """
    Exception raised when either the Environment, the NAFAgent or both have not been initialized
    yet, and the user tries to execute a training by calling the run_training() method.
    """
    message = 'The configuration for the training is incomplete. Either the Environment, the ' \
              'NAF Agent or both are not yet initialized. The environment can be initialized via the ' \
              'initialize_environment() method, and the agent can be initialized via the ' \
              'initialize_naf_agent() method'

    def __init__(self, message: Optional[str] = None) -> None:
        if message:
            self.message = message
        FrameworkException.__init__(self, self.message)
