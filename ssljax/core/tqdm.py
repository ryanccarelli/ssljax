"""
Wrapper for the tqdm library to prevent log files filling up with status bar
updates and removes the en the logged messages so that their format is better
for files. It removes the end carriage added by tqdm's writer.

The TQDM handlers were taken from [AllenNLP's tqdm handler](https://github.com/allenai/allennlp/blob/main/allennlp/common/tqdm.py)

No changes were made to this file and thus it falls under the Apache 2.0 License.
"""
import logging
import sys
from time import time
from typing import Optional

from ssljax.core import log_util as common_logging

try:
    SHELL = str(type(get_ipython()))  # type:ignore # noqa: F821
except:  # noqa: E722
    SHELL = ""

if "zmqshell.ZMQInteractiveShell" in SHELL:
    from tqdm import tqdm_notebook as _tqdm
else:
    from tqdm import tqdm as _tqdm

# This is necessary to stop tqdm from hanging
# when exceptions are raised inside iterators.
# It should have been fixed in 4.2.1, but it still
# occurs.
_tqdm.monitor_interval = 0

logger = logging.getLogger("tqdm")
logger.propagate = False


def replace_cr_with_newline(message: str) -> str:
    """
    TQDM and requests use carriage returns to get the training line to update
    for each batch without adding more lines to the terminal output. Displaying
    those in a file won't work correctly, so we'll just make sure that each
    batch shows up on its one line.
    """
    # In addition to carriage returns, nested progress-bars will contain extra new-line
    # characters and this special control sequence which tells the terminal to move the
    # cursor one line up.
    message = message.replace("\r", "").replace("\n", "").replace("", "")
    if message and message[-1] != "\n":
        message += "\n"
    return message


class TqdmToLogsWriter(object):
    """
    Log writer object that rate limits the number of messages that can be
    written to the progress bar. Also formats the message and removes carriage
    characters that will mess up the log files.
    """

    def __init__(self):
        self.last_message_written_time = 0.0

    def write(self, message: str):
        """
        Write the message.

        Args:
            message (str): The message to write.

        Returns: None

        """
        file_friendly_message: Optional[str] = None
        if common_logging.FILE_FRIENDLY_LOGGING:
            file_friendly_message = replace_cr_with_newline(message)
            if file_friendly_message.strip():
                sys.stderr.write(file_friendly_message)
        else:
            sys.stderr.write(message)

        # Every 10 seconds we also log the message.
        now = time()
        if now - self.last_message_written_time >= 10 or "100%" in message:
            if file_friendly_message is None:
                file_friendly_message = replace_cr_with_newline(message)
            for message in file_friendly_message.split("\n"):
                message = message.strip()
                if len(message) > 0:
                    logger.info(message)
                    self.last_message_written_time = now

    def flush(self):
        sys.stderr.flush()


class Tqdm:
    @staticmethod
    def tqdm(*args, **kwargs):
        # Use a slower interval when FILE_FRIENDLY_LOGGING is set.
        default_mininterval = 2.0 if common_logging.FILE_FRIENDLY_LOGGING else 0.1

        new_kwargs = {
            "file": TqdmToLogsWriter(),
            "mininterval": default_mininterval,
            **kwargs,
        }

        return _tqdm(*args, **new_kwargs)

    @staticmethod
    def set_lock(lock):
        _tqdm.set_lock(lock)

    @staticmethod
    def get_lock():
        return _tqdm.get_lock()
