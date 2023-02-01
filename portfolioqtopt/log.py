import sys
import typing

from loguru import logger


def enable(
    level: str,
    file_dir: typing.Optional[str] = None,
    file_level: typing.Optional[str] = None,
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
    logger.enable("portfolioqtopt")
    logger.remove()
    if file_dir:
        file_path = "/".join([f"{file_dir}", "{time}.log"])
        if file_level is None:
            file_level = "INFO"
        logger.add(file_path, level=file_level)
    logger.add(sys.stderr, level=level)
