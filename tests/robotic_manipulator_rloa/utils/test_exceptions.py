import pytest
from mock import MagicMock, patch
from typing import Union

from robotic_manipulator_rloa.utils.exceptions import (
    FrameworkException,
    InvalidManipulatorFile,
    InvalidHyperParameter,
    InvalidEnvironmentParameter,
    InvalidNAFAgentParameter,
    EnvironmentNotInitialized,
    NAFAgentNotInitialized,
    MissingWeightsFile,
    ConfigurationIncomplete
)


def test_framework_exception() -> None:
    """Test for the base exception class FrameworkException"""
    exception = FrameworkException(message='test_exception')
    assert exception.message == 'test_exception'
    assert str(exception) == 'FrameworkException: test_exception'
    exception.set_message('test msg')
    assert exception.message == 'test msg'


@pytest.mark.parametrize('msg', [None, 'error_msg'])
def test_invalid_manipulator_file(msg: Union[str, None]) -> None:
    """Test for the InvalidManipulatorFile exception class"""
    if msg:
        exception = InvalidManipulatorFile(msg)
        assert exception.message == msg
    else:
        exception = InvalidManipulatorFile()
        assert exception.message == 'The URDF/SDF file received is not valid'


@pytest.mark.parametrize('msg', [None, 'error_msg'])
def test_invalid_hyperparameter(msg: Union[str, None]) -> None:
    """Test for the InvalidHyperParameter exception class"""
    if msg:
        exception = InvalidHyperParameter(msg)
        assert exception.message == msg
    else:
        exception = InvalidHyperParameter()
        assert exception.message == 'The hyperparameter received is not valid'


@pytest.mark.parametrize('msg', [None, 'error_msg'])
def test_invalid_environment_parameter(msg: Union[str, None]) -> None:
    """Test for the InvalidEnvironmentParameter exception class"""
    if msg:
        exception = InvalidEnvironmentParameter(msg)
        assert exception.message == msg
    else:
        exception = InvalidEnvironmentParameter()
        assert exception.message == 'The Environment parameter received is not valid'


@pytest.mark.parametrize('msg', [None, 'error_msg'])
def test_invalid_nafagent_parameter(msg: Union[str, None]) -> None:
    """Test for the InvalidNAFAgentParameter exception class"""
    if msg:
        exception = InvalidNAFAgentParameter(msg)
        assert exception.message == msg
    else:
        exception = InvalidNAFAgentParameter()
        assert exception.message == 'The NAF Agent parameter received is not valid'


@pytest.mark.parametrize('msg', [None, 'error_msg'])
def test_environment_not_initialized(msg: Union[str, None]) -> None:
    """Test for the EnvironmentNotInitialized exception class"""
    if msg:
        exception = EnvironmentNotInitialized(msg)
        assert exception.message == msg
    else:
        exception = EnvironmentNotInitialized()
        assert exception.message == 'The Environment is not yet initialized. The environment can be initialized ' \
                                    'via the initialize_environment() method'


@pytest.mark.parametrize('msg', [None, 'error_msg'])
def test_nafagent_not_initialized(msg: Union[str, None]) -> None:
    """Test for the NAFAgentNotInitialized exception class"""
    if msg:
        exception = NAFAgentNotInitialized(msg)
        assert exception.message == msg
    else:
        exception = NAFAgentNotInitialized()
        assert exception.message == 'The NAF Agent is not yet initialized. The agent can be initialized via the ' \
                                    'initialize_naf_agent() method'


@pytest.mark.parametrize('msg', [None, 'error_msg'])
def test_missing_weights_file(msg: Union[str, None]) -> None:
    """Test for the MissingWeightsFile exception class"""
    if msg:
        exception = MissingWeightsFile(msg)
        assert exception.message == msg
    else:
        exception = MissingWeightsFile()
        assert exception.message == 'The weight file provided does not exist'


@pytest.mark.parametrize('msg', [None, 'error_msg'])
def test_configuration_incomplete(msg: Union[str, None]) -> None:
    """Test for the ConfigurationIncomplete exception class"""
    if msg:
        exception = ConfigurationIncomplete(msg)
        assert exception.message == msg
    else:
        exception = ConfigurationIncomplete()
        assert exception.message == 'The configuration for the training is incomplete. Either the Environment, ' \
                                    'the NAF Agent or both are not yet initialized. The environment can be initialized ' \
                                    'via the initialize_environment() method, and the agent can be initialized via ' \
                                    'the initialize_naf_agent() method'
