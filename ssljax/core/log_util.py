"""
Logging utility functions to create the global logging handlers.

This was adapted from [AllenNLP's Prepare Logging](https://github.com/allenai/allennlp/blob/main/allennlp/common/logging.py)

Our changes are as follows:

* Different formatting for logging messages
* Removed the logger class

"""
import logging
import os
import sys
from logging import Filter
from os import PathLike
from typing import Union

logger = logging.getLogger(__name__)

FILE_FRIENDLY_LOGGING: bool = False
"""
If this flag is set to `True`, we add newlines to tqdm output, even on an
interactive terminal, and we slow down tqdm's output to only once every 10
seconds. By default, it is set to `False`.
"""


class ErrorFilter(Filter):
    """
    Filters out everything that is at the ERROR level or higher. This is meant to be used
    with a stdout handler when a stderr handler is also configured. That way ERROR
    messages aren't duplicated.
    """

    def filter(self, record):
        return record.levelno < logging.ERROR


def prepare_global_logging(
    serialization_dir: Union[str, PathLike],
    rank: int = 0,
    world_size: int = 1,
    log_file_name: str = "out",
) -> None:
    """
    Prepare the global logging handlers.

    If you want to enable debugging logging, set `os.environ["SSLJAX_DEBUG"]` to
        any string value (because bools cannot be stored in th os environment).

    To set the level set `os.environ["SSLJAX_LOG_LEVEL"]` to the string name of
        the logging level.

    Args:
        serialization_dir (`Union[str,PathLike]`): The directory where the logs
            files will be written too.
        rank (`int`, default=`0`): The current rank of the process that called
            this function.
        world_size (`int`, default=`1`): Number of processes.
        log_file_name (`str`, default=`out`): The file name of the output log
            file.

    Returns: None
    """

    root_logger = logging.getLogger()

    # Define three different logging formats, the purpose of this is to make
    # console messages brief while file and errors are verbose.
    console_format = "[%(levelname)8s] %(message)s"
    file_format = "[%(asctime)s - %(levelname)8s - %(name)s] %(message)s"
    error_format = "[%(asctime)s - %(levelname)8s - %(name)s] %(message)s"

    # Create the handlers. If the world size is greater than 1, than there are
    # more than 1 process running. So we need to mark that in the logging.
    if world_size == 1:
        log_file = os.path.join(serialization_dir, f"{log_file_name}.log")
    else:
        log_file = os.path.join(serialization_dir, f"{logging}_worker{rank}.log")
        console_format = f"{rank} | {console_format}"
        file_format = f"{rank} | {file_format}"
        error_format = f"{rank} | {error_format}"

    console_format = logging.Formatter(fmt=console_format)
    error_format = logging.Formatter(fmt=error_format, datefmt="%Y-%m-%d %H:%M:%S")
    file_format = logging.Formatter(fmt=file_format, datefmt="%Y-%m-%d %H:%M:%S")

    file_handler = logging.FileHandler(log_file)
    stderr_handler = logging.StreamHandler(sys.stderr)

    file_handler.setFormatter(file_format)
    stderr_handler.setFormatter(error_format)

    stdout_handler = logging.StreamHandler(sys.stdout)
    if os.environ.get("SSLJAX_DEBUG"):
        LEVEL = logging.DEBUG

        # For debugging, set the formatter to verbose.
        stdout_handler.setFormatter(file_format)
    else:
        stdout_handler.setFormatter(console_format)
        level_name = os.environ.get("SSLJAX_LOG_LEVEL", "INFO")
        LEVEL = logging._nameToLevel.get(level_name, logging.INFO)

    # Remove the already set handlers in root logger.
    # Not doing this will result in duplicate log messages
    root_logger.handlers.clear()

    # Set the levels for the handlers.
    file_handler.setLevel(LEVEL)
    stdout_handler.setLevel(LEVEL)
    stderr_handler.setLevel(logging.ERROR)
    root_logger.setLevel(LEVEL)

    # Make sure errors only go to stderr
    stdout_handler.addFilter(ErrorFilter())

    # put all the handlers on the root logger
    root_logger.addHandler(file_handler)
    if rank == 0:
        root_logger.addHandler(stdout_handler)
        root_logger.addHandler(stderr_handler)

    # write uncaught exceptions to the logs
    def excepthook(exctype, value, traceback):
        # For a KeyboardInterrupt, call the original exception handler.
        if issubclass(exctype, KeyboardInterrupt):
            sys.__excepthook__(exctype, value, traceback)
            return
        root_logger.critical("Uncaught exception", exc_info=(exctype, value, traceback))

    sys.excepthook = excepthook

    # Avoid circular imports by doing this
    from ssljax.core.tqdm import logger as tqdm_logger

    tqdm_logger.handlers.clear()
    tqdm_logger.addHandler(file_handler)
