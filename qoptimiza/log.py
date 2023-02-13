import enum
import sys
import typing

from loguru import logger


class LevelName(enum.Enum):
    TRACE = 5
    DEBUG = 10
    INFO = 20
    SUCCESS = 25
    WARNING = 30
    ERROR = 40
    CRITICAL = 50


def enable(
    level: LevelName,
    file_dir: typing.Optional[str] = None,
    file_level: typing.Optional[LevelName] = None,
) -> None:
    """Enable logging message.

    The logger of the library is deactivated so the logging function are no-op inside
    the library. When you wish to see the logs, you can activate it with the
    :func:`enable` function.

    Args:
        level (str): The sys.stderr log level.
        file_dir (typing.Optional[str], optional): A directory where the log file will
            be written. Defaults to None.
        file_level (typing.Optional[str], optional): THe log level of the file handler.
            Defaults to None.

    """
    logger.enable("qoptimiza")
    logger.remove()
    if file_dir:
        file_path = "/".join([f"{file_dir}", "{time}.log"])
        if file_level is None:
            file_level = LevelName.INFO
        logger.add(file_path, level=file_level.value)
    logger.add(sys.stderr, level=level.value)
