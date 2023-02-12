import logging
from datetime import datetime
from logging import LogRecord
from logging.config import dictConfig
from logging.handlers import RotatingFileHandler


class Logger:  # pylint: disable=too-many-instance-attributes
    """ Stores and processes the logs """

    @staticmethod
    def set_logger_setup() -> None:
        """
        Sets the logger setup with a predefined configuration.
        """

        log_config_dict = Logger.generate_logging_config_dict()
        dictConfig(log_config_dict)
        rotating_file_handler = RotatingFileHandler(filename='training_logs.log', mode='a', maxBytes=50000000,
                                                    backupCount=10, encoding='utf-8')
        rotating_file_handler.setFormatter(logging.Formatter(
            '"%(levelname)s"|"{datetime}"|%(message)s'.format(datetime=datetime.utcnow().strftime('%Y-%m-%dT%H:%M:%S.%fZ'))
        ))
        rotating_file_handler.setLevel(logging.INFO)
        logger = get_global_logger()
        logger.addHandler(rotating_file_handler)
        logger.setLevel(20)

    @staticmethod
    def generate_logging_config_dict() -> dict:
        """
        Generates the configuration dictionary that is used to configure the logger.
        Returns:
            Configuration dictionary.
        """
        return {
            'version': 1,
            'disable_existing_loggers': False,
            'formatters': {
                'custom_formatter': {
                    '()': CustomFormatter,
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


def get_global_logger() -> logging.Logger:
    """
    Getter for the logger.
    Returns:
        Logger instance to be used on the framework.
    """
    return logging.getLogger(__name__)


class CustomFormatter(logging.Formatter):

    def __init__(self, dateformat: str = None):
        """
        CustomFormatter for the logger.
        """
        super().__init__()
        self.dateformat = dateformat

    def format(self, record: LogRecord) -> str:
        """
        Formats the provided LogRecord instance.
        Returns:
            Formatted LogRecord as string.
        """
        # Set format and colors
        grey = "\033[38;20m"
        green = "\033[32;20m"
        yellow = "\033[33;20m"
        red = "\033[31;20m"
        bold_red = "\033[31;1m"
        reset = "\033[0m"
        format_ = '[%(levelname)-8s] - {datetime} - %(message)s'.format(
            datetime=datetime.now().astimezone().strftime('%Y-%m-%dT%H:%M:%S.%f%z')
        )

        self.FORMATS = {
            logging.DEBUG: green + format_ + reset,
            logging.INFO: grey + format_ + reset,
            logging.WARNING: yellow + format_ + reset,
            logging.ERROR: red + format_ + reset,
            logging.CRITICAL: bold_red + format_ + reset
        }

        log_format = self.FORMATS.get(record.levelno)

        formatter = logging.Formatter(log_format, datefmt=self.dateformat)
        return formatter.format(record)
