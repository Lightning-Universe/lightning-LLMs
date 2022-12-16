import time
from unittest.mock import Mock

import lightning as L
import pytest

from lit_llms.tensorboard import DriveTensorBoardLogger


@pytest.mark.parametrize("refresh_time", [0.25, 0.5])
def test_log_metrics(tmpdir, refresh_time):
    class CustomDriveTensorBoardLogger(DriveTensorBoardLogger):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._upload_to_storage = Mock()

    logger = CustomDriveTensorBoardLogger(
        save_dir=tmpdir, drive=L.app.storage.Drive("lit://dummy"), refresh_time=refresh_time
    )

    assert logger.timestamp is None
    logger._upload_to_storage.assert_not_called()
    logger.log_metrics(metrics={"a": 1}, step=0)
    logger._upload_to_storage.assert_called_once()
    time.sleep(0.1)
    logger.log_metrics(metrics={"a": 2}, step=2)
    logger._upload_to_storage.assert_called_once()

    time.sleep(refresh_time)
    logger.log_metrics(metrics={"a": 3}, step=3)
    assert logger._upload_to_storage.call_count == 2

    time.sleep(0.1)
    logger.log_metrics(metrics={"a": 4}, step=4)
    assert logger._upload_to_storage.call_count == 2

    time.sleep(refresh_time)
    logger.log_metrics(metrics={"a": 5}, step=5)
    assert logger._upload_to_storage.call_count == 3
