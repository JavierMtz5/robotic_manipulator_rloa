import logging

import pytest
from mock import MagicMock, patch

from robotic_manipulator_rloa.utils.logger import Logger, get_global_logger, CustomFormatter


@patch('robotic_manipulator_rloa.utils.logger.Logger.generate_logging_config_dict')
@patch('robotic_manipulator_rloa.utils.logger.dictConfig')
@patch('robotic_manipulator_rloa.utils.logger.RotatingFileHandler')
@patch('robotic_manipulator_rloa.utils.logger.logging.Formatter')
@patch('robotic_manipulator_rloa.utils.logger.get_global_logger')
@patch('robotic_manipulator_rloa.utils.logger.datetime')
def test_logger(mock_datetime: MagicMock,
                mock_get_global_logger: MagicMock,
                mock_formatter: MagicMock,
                mock_rotating_file_handler: MagicMock,
                mock_dictConfig: MagicMock,
                mock_generate_config_dict: MagicMock) -> None:
    """Test for the Logger class"""
    fake_utcnow, fake_strftime = MagicMock(), MagicMock()
    mock_datetime.utcnow.return_value = fake_utcnow
    fake_utcnow.strftime.return_value = 'datetime'
    fake_config_dict = mock_generate_config_dict.return_value
    mock_dictConfig.return_value = None
    fake_rotating_file_handler, fake_formatter = MagicMock(), MagicMock()
    mock_rotating_file_handler.return_value = fake_rotating_file_handler
    mock_formatter.return_value = fake_formatter
    fake_rotating_file_handler.setFormatter.return_value = None
    fake_rotating_file_handler.setLevel.return_value = None
    fake_logger = MagicMock()
    mock_get_global_logger.return_value = fake_logger
    fake_logger.addHandler.return_value = None
    fake_logger.setLevel.return_value = None

    Logger.set_logger_setup()

    mock_generate_config_dict.assert_any_call()
    mock_dictConfig.assert_any_call(fake_config_dict)
    mock_rotating_file_handler.assert_any_call(filename='training_logs.log', mode='a', maxBytes=50000000,
                                               backupCount=10, encoding='utf-8')
    mock_formatter.assert_any_call('"%(levelname)s"|"datetime"|%(message)s')
    fake_rotating_file_handler.setFormatter.assert_any_call(fake_formatter)
    fake_rotating_file_handler.setLevel.assert_any_call(logging.INFO)
    mock_get_global_logger.assert_any_call()
    fake_logger.addHandler.assert_any_call(fake_rotating_file_handler)
    fake_logger.setLevel.assert_any_call(20)


@patch('robotic_manipulator_rloa.utils.logger.CustomFormatter')
def test_logger__generate_logging_config_dict(mock_custom_formatter: MagicMock) -> None:
    """Test for the generate_logging_config_dict() method of the Logger class"""
    output = {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'custom_formatter': {
                    '()': mock_custom_formatter,
                    'dateformat': '%Y-%m-%dT%H:%M:%S.%06d%z'
                },
            },
            'handlers': {
                'debug_console_handler': {
                    'level': 'NOTSET',
                    'formatter': 'custom_formatter',
                    'class': 'logging.StreamHandler',
                    'stream': 'ext://sys.stdout',
                }
            },
            'loggers': {
                '': {
                    'handlers': ['debug_console_handler'],
                    'level': 'NOTSET',
                },
            }
        }
    assert Logger.generate_logging_config_dict() == output


@patch('robotic_manipulator_rloa.utils.logger.logging.getLogger')
def test_get_global_logger(mock_get_logger: MagicMock) -> None:
    """Test for the get_global_logger() method"""
    mock_get_logger.return_value = 'logger'
    assert get_global_logger() == 'logger'
    mock_get_logger.assert_any_call('robotic_manipulator_rloa.utils.logger')


@patch('robotic_manipulator_rloa.utils.logger.logging.Formatter')
@patch('robotic_manipulator_rloa.utils.logger.datetime')
def test_customformatter(mock_datetime: MagicMock,
                         mock_formatter: MagicMock) -> None:
    """Test for the CustomFormatter class"""
    custom_formatter = CustomFormatter(dateformat='dateformat')
    assert custom_formatter.dateformat == 'dateformat'

    # ================== TEST FOR format() method =============================

    fake_formatter, fake_now, fake_astimezone = MagicMock(), MagicMock(), MagicMock()
    fake_input_record = MagicMock(levelno=logging.DEBUG)
    mock_formatter.return_value = fake_formatter
    fake_formatter.format.return_value = 'formatted_record'
    mock_datetime.now.return_value = fake_now
    fake_now.astimezone.return_value = fake_astimezone
    fake_astimezone.strftime.return_value = 'datetime'

    assert custom_formatter.format(fake_input_record) == 'formatted_record'

    mock_formatter.assert_any_call("\033[32;20m" + "[%(levelname)-8s] - datetime - %(message)s" + "\033[0m",
                                   datefmt='dateformat')


