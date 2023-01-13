import concurrent.futures
import os
import sys
from functools import partial
from pathlib import Path
from subprocess import Popen
from time import time
from typing import Any, List, Mapping, Optional, Type
from uuid import uuid4

import fsspec
import lightning as L
from fsspec.implementations.local import LocalFileSystem
from lightning.app.utilities.exceptions import ExitAppException


class DriveTensorBoardLogger(L.pytorch.loggers.TensorBoardLogger):
    def __init__(self, *args: Any, drive: L.app.storage.Drive, refresh_time: int = 5, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.timestamp: Optional[float] = None
        self.drive = drive
        self.refresh_time = refresh_time

    @L.pytorch.utilities.rank_zero.rank_zero_only
    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        super().log_metrics(metrics, step)
        if self.timestamp is None:
            self._upload_to_storage()
            self.timestamp = time()
        elif (time() - self.timestamp) > self.refresh_time:
            self._upload_to_storage()
            self.timestamp = time()

    def _upload_to_storage(self) -> None:
        fs = L.app.storage.path._filesystem()
        fs.invalidate_cache()

        source_path = Path(self.log_dir).resolve()
        destination_path = self.drive._to_shared_path(self.log_dir, component_name=self.drive.component_name)

        src = [file for file in source_path.rglob("*") if file.is_file()]
        dst = [destination_path / file.relative_to(source_path) for file in src]

        with concurrent.futures.ThreadPoolExecutor(4) as executor:
            results = executor.map(partial(self._copy, fs=fs), src, dst)

        # Raise the first exception found
        exception = next((e for e in results if isinstance(e, Exception)), None)
        if exception:
            raise Exception

    @staticmethod
    def _copy(src_path: Path, dst_path: Path, fs: fsspec.AbstractFileSystem) -> Optional[Exception]:
        try:
            # NOTE: S3 does not have a concept of directories, so we do not need to create one.
            if isinstance(fs, LocalFileSystem):
                fs.makedirs(str(dst_path.parent), exist_ok=True)

            fs.put(str(src_path), str(dst_path), recursive=False)

            # Don't delete tensorboard logs.
            if "events.out.tfevents" not in str(src_path):
                os.remove(str(src_path))

        except Exception as e:
            # Return the exception so that it can be handled in the main thread
            return e
        return None


class TensorBoardWork(L.app.LightningWork):
    def __init__(self, *args: Any, drive: L.app.storage.Drive, **kwargs: Any):
        super().__init__(
            *args,
            parallel=True,
            cloud_build_config=L.BuildConfig(requirements=["tensorboard"]),
            **kwargs,
        )

        self.drive = drive

    def run(self) -> None:

        use_localhost = not L.app.utilities.cloud.is_running_in_cloud()

        local_folder = f"./tensorboard_logs/{uuid4()}"

        os.makedirs(local_folder, exist_ok=True)

        # Note: Used tensorboard built-in sync methods but it doesn't seem to work.
        cmd = f"tensorboard --logdir={local_folder} --host {self.host} --port {self.port}"

        if use_localhost:
            # installs tensorboard ONLY in the process it needs to be in
            # necessary because local build configs are not yet supported
            cmd = (
                "GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1 GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1 "
                + sys.executable
                + " -m pip install tensorboard && "
                + cmd
            )

        self._process = Popen(cmd, shell=True, env=os.environ)
        print(f"Running Tensorboard on {self.host}:{self.port}")

        fs = L.app.storage.path._filesystem()
        root_folder = str(self.drive.drive_root)

        while True:
            fs.invalidate_cache()
            for dir, _, files in fs.walk(root_folder):
                for filepath in files:
                    if "events.out.tfevents" not in filepath:
                        continue
                    source_path = os.path.join(dir, filepath)
                    target_path = os.path.join(dir, filepath).replace(root_folder, local_folder)
                    if use_localhost:
                        parent = Path(target_path).resolve().parent
                        if not parent.exists():
                            parent.mkdir(exist_ok=True, parents=True)
                    fs.get(source_path, str(Path(target_path).resolve()))

    def on_exit(self) -> None:
        assert self._process
        self._process.kill()


class MultiNodeLightningTrainerWithTensorboard(L.LightningFlow):
    def __init__(
        self,
        work_cls: Type[L.LightningWork],
        num_nodes: int,
        cloud_compute: L.CloudCompute,
        quit_tb_with_training: bool = True,
    ):
        super().__init__()
        tb_drive = L.app.storage.Drive("lit://tb_drive")
        self.tensorboard_work = TensorBoardWork(drive=tb_drive)
        self.multinode = L.app.components.LightningTrainerMultiNode(
            work_cls,
            num_nodes=num_nodes,
            cloud_compute=cloud_compute,
            tb_drive=tb_drive,
        )
        self.quit_tb_with_training = quit_tb_with_training

    def run(self, *args: Any, **kwargs: Any) -> None:
        if self.quit_tb_with_training and all([w.has_succeeded for w in self.multinode.ws]):
            self.tensorboard_work.stop()
            raise ExitAppException

        self.tensorboard_work.run()
        self.multinode.run(*args, **kwargs)

    def configure_layout(self) -> List[Mapping[str, str]]:
        return [{"name": "Training Logs", "content": self.tensorboard_work.url}]
