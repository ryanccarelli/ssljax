import os

import pytest
from ssljax.core import log_util as common_logging
from ssljax.core.tqdm import Tqdm


def test_reset_tqdm_logger_handlers(tmpdir):
    serialization_dir_a = os.path.join(tmpdir, "test_a")
    os.makedirs(serialization_dir_a, exist_ok=True)
    common_logging.prepare_global_logging(
        serialization_dir_a, log_file_name="test_reset_tqdm_logger_handlers"
    )

    serialization_dir_b = os.path.join(tmpdir, "test_b")
    os.makedirs(serialization_dir_b, exist_ok=True)
    common_logging.prepare_global_logging(
        serialization_dir_b, log_file_name="test_reset_tqdm_logger_handlers"
    )

    # Use range(1) to make sure there should be only 2 lines in the file (0% and 100%)
    for _ in Tqdm.tqdm(range(1)):
        pass

    with open(
        os.path.join(serialization_dir_a, "test_reset_tqdm_logger_handlers.log"), "r"
    ) as f:
        assert len(f.readlines()) == 0
    with open(
        os.path.join(serialization_dir_b, "test_reset_tqdm_logger_handlers.log"), "r"
    ) as f:
        assert len(f.readlines()) == 2
