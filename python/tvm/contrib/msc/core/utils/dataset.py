# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=unused-argument
"""tvm.contrib.msc.core.utils.dataset"""

import os
import shutil
import json
from typing import List, Union, Dict, Any, Tuple
import numpy as np

import tvm
from .arguments import load_dict
from .info import cast_array, is_array
from .namespace import MSCFramework


def format_datas(datas: Union[List[Any], Dict[str, Any]], names: List[str], style="dict") -> Any:
    """Format datas to style format

    Parameters
    ----------
    datas:
        The source datas.
    names: list<str>
        The data names.
    style: str
        The style of format, dict|list.

    Returns
    -------
    datas:
        The formated datas.
    """

    if isinstance(datas, (list, tuple, tvm.ir.container.Array)):
        assert len(datas) == len(names), "datas({}) mismatch with names {}".format(
            len(datas), names
        )
        datas = dict(zip(names, datas))
    if not isinstance(datas, dict):
        assert len(names) == 1, "Expect 1 names, get " + str(names)
        datas = {names[0]: datas}
    elif len(datas) > len(names):
        datas = {n: datas[n] for n in datas}
    assert all(is_array(d) for d in datas.values()), "Expected all tensors as array like"
    if style == "dict":
        return datas
    if style == "list":
        return [datas[n] for n in names]
    raise TypeError("Unexpected style " + str(style))


def random_data(
    info: Union[List, Tuple, dict],
    framework: str = MSCFramework.MSC,
    device: str = "cpu",
    max_val: int = None,
) -> Any:
    """Create random data from info

    Parameters
    ----------
    info: list| tuple| dict
        The data info.
    framework: str
        The framework.
    device: str
        The device.
    """

    if isinstance(info, (tuple, list)):
        if len(info) == 1:
            info = {"name": "data", "shape": info[0], "dtype": "float32"}
        elif len(info) == 2:
            info = {"name": "data", "shape": info[0], "dtype": info[1]}
        elif len(info) == 3:
            info = {"name": info[0], "shape": info[1], "dtype": info[2]}
        else:
            raise Exception("Unexpected info " + str(info))
    assert isinstance(info, dict) and all(
        key in info for key in ["shape", "dtype"]
    ), "shape and dtype should be given to create randome data"
    if info["dtype"] in ("int32", "int64"):
        if max_val is None:
            data = np.zeros(info["shape"]).astype(info["dtype"])
        else:
            data = np.random.randint(0, high=max_val, size=info["shape"]).astype(info["dtype"])
    elif info["dtype"] == "bool":
        data = np.random.rand(*info["shape"]).astype("float32")
        data = np.where(data >= 0.5, True, False)
    else:
        data = np.random.rand(*info["shape"]).astype(info["dtype"])
        if max_val is not None:
            data *= max_val
    return cast_array(data, framework, device=device)


class BaseDataLoader(object):
    """Basic dataset loader for MSC

    Parameters
    ----------
    folder: string
        The dataset folder path.
    start: int
        The start position.
    end: int
        The end position.
    """

    def __init__(self, folder: str, start: int = 0, end: int = -1):
        self._folder = folder
        self._start = start
        self._current = 0
        assert os.path.isdir(folder), "Dataset {} is not folder".format(folder)
        self._info = load_dict(os.path.join(folder, "datas_info.json"))
        if end == -1:
            self._end = self._info["num_datas"]
        else:
            self._end = min(end, self._info["num_datas"])

    def __str__(self):
        return "<{}> @ {}".format(self.__class__.__name__, self._folder)

    def __getitem__(self, idx):
        if idx + self._start >= self._end:
            raise StopIteration("Reach End")
        return self._load_batch(idx)

    def __next__(self):
        if self._current + self._start >= self._end:
            raise StopIteration("Reach End")
        batch = self._load_batch(self._current)
        self._current += 1
        return batch

    def __len__(self):
        return self._end - self._start

    def reset(self):
        self._current = 0

    def has_data(self, name: str, index: int) -> bool:
        """Check if data exist.

        Parameters
        -------
        name: str
            The name of the data.
        index: int
            The index of the data.

        Returns
        -------
        has_data: bool
           Whether the data can be load.
        """

        info = self._data_info(name)
        if not info:
            return False
        save_name = info.get("save_name", name)
        f_path = os.path.join(self._folder, save_name, "batch_{}.bin".format(self._start + index))
        return os.path.isfile(f_path)

    def load_data(self, name: str, index: int) -> np.ndarray:
        """Load data by name.

        Parameters
        -------
        name: str
            The name of the data.
        index: int
            The index of the data.

        Returns
        -------
        data: np.ndarray
           The loaded data.
        """

        return self._load_data(name, index, self._data_info(name))

    def _load_data(self, name: str, index: int, info: dict) -> np.ndarray:
        """Load data from file.

        Parameters
        -------
        name: str
            The name of the data.
        index: int
            The index of the data.
        info: dict
            The info of the data.

        Returns
        -------
        data: np.ndarray
           The loaded data.
        """

        save_name = info.get("save_name", name)
        f_path = os.path.join(self._folder, save_name, "batch_{}.bin".format(self._start + index))
        assert os.path.isfile(f_path), "Can not find data file " + str(f_path)
        return np.fromfile(f_path, dtype=info["dtype"]).reshape(info["shape"])

    def _load_batch(self, index: int) -> Any:
        """Get batch data

        Parameters
        -------
        index: int
            The index for the batch.

        Returns
        -------
        batch: Any
           The batch data.
        """

        raise NotImplementedError("_load_batch is not implemented for BaseDataLoader")

    def _data_info(self, name: str) -> dict:
        """Get info of data

        Parameters
        -------
        name: str
            The name of data.

        Returns
        -------
        info: dict
           The info of data.
        """

        raise NotImplementedError("_data_info is not implemented for BaseDataLoader")

    @property
    def num_datas(self):
        return self.info["num_datas"]

    @property
    def folder(self):
        return self._folder

    @property
    def info(self):
        return self._info


class SimpleDataLoader(BaseDataLoader):
    """Dataset Loader for simple datas"""

    def _load_batch(self, index: int) -> Any:
        """Get batch data

        Parameters
        -------
        index: int
            The index for the batch.

        Returns
        -------
        batch: Any
           The batch data.
        """

        assert "datas" in self._info, "datas shoule be given to load batch"
        return {n: self._load_data(n, index, i) for n, i in self._info["datas"].items()}

    def _data_info(self, name: str) -> dict:
        """Get info of data

        Parameters
        -------
        name: str
            The name of data.

        Returns
        -------
        info: dict
           The info of data.
        """

        return self._info["datas"].get(name)


class IODataLoader(BaseDataLoader):
    """Dataset Loader for Input/Output datas"""

    def _load_batch(self, index: int) -> Any:
        """Get batch data

        Parameters
        -------
        index: int
            The index for the batch.

        Returns
        -------
        batch: Any
           The batch data.
        """

        if "inputs" in self._info:
            inputs = {n: self._load_data(n, index, i) for n, i in self._info["inputs"].items()}
        else:
            inputs = {}
        if "outputs" in self._info:
            outputs = {n: self._load_data(n, index, i) for n, i in self._info["outputs"].items()}
        else:
            outputs = {}
        return inputs, outputs

    def _data_info(self, name: str) -> dict:
        """Get info of data

        Parameters
        -------
        name: str
            The name of data.

        Returns
        -------
        info: dict
           The info of data.
        """

        if name in self._info["inputs"]:
            return self._info["inputs"][name]
        return self._info["outputs"].get(name)


class BaseDataSaver(object):
    """Dataset Saver for MSC

    Parameters
    ----------
    folder: string
        The dataset folder path.
    options: dict
        The extra options for the data saver
    start: int
        The start position.
    max_size: int
        The max size for datas.
    """

    def __init__(
        self,
        folder: str,
        options: dict = None,
        start: int = 0,
        max_size: int = -1,
    ):
        if os.path.isdir(folder):
            shutil.rmtree(folder)
        os.mkdir(folder)
        self._folder = folder
        self._start = start
        self._max_size = max_size
        self._current = 0
        assert os.path.isdir(folder), "Dataset {} is not folder".format(folder)
        self._info = self.setup(options)

    def setup(self, options: dict):
        return {"num_datas": 0}

    def __str__(self):
        return "<{}> @ {}".format(self.__class__.__name__, self._folder)

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.finalize()

    def finalize(self):
        """Finalize the saver"""

        self._info["num_datas"] = self._current
        with open(os.path.join(self._folder, "datas_info.json"), "w") as f:
            f.write(json.dumps(self._info, indent=2))

    def is_finalized(self) -> bool:
        """Check if the saver is finalized

        Returns
        -------
        is_finalized: bool
           Whether the saver is finalized.
        """

        return os.path.isfile(os.path.join(self._folder, "datas_info.json"))

    def reset(self):
        self._current = 0

    def _save_data(self, index: int, name: str, data: np.ndarray, collect: str) -> str:
        """Save data to file.

        Parameters
        -------
        index: int
            The index
        name: str
            The name of the data.
        data: np.ndarray
           The data to be saved.
        collect: str
            The collect of data.

        Returns
        -------
        data_path: str
           The folder that data saved to.
        """

        data = cast_array(data)
        save_name = name.replace("/", "_").replace(":", "_")
        sub_folder = f_path = os.path.join(self._folder, save_name)
        if not os.path.isdir(sub_folder):
            os.mkdir(sub_folder)
        f_path = os.path.join(sub_folder, "batch_{}.bin".format(self._start + index))
        ref_info = self._info[collect]
        # TODO(mengtong): support dynamic datas shape
        if name in ref_info:
            assert (
                ref_info[name]["dtype"] == data.dtype.name
            ), "dtype {} mismatch with saved {}".format(data.dtype.name, ref_info[name]["dtype"])
            assert ref_info[name]["shape"] == list(
                data.shape
            ), "shape {} mismatch with saved {}".format(data.shape, ref_info[name]["shape"])
        else:
            ref_info[name] = {
                "shape": list(data.shape),
                "dtype": data.dtype.name,
                "bytes": data.size * data.itemsize,
                "save_name": save_name,
            }
        data.tofile(f_path)
        return sub_folder

    def _save_batch(self, *args, **kwargs) -> dict:
        """Save a batch data"""

        raise NotImplementedError("_save_batch is not implemented for BaseDataSaver")

    @property
    def num_datas(self):
        if self.is_finalized():
            return self.info["num_datas"]
        return self._current

    @property
    def folder(self):
        return self._folder

    @property
    def info(self):
        return self._info


class SimpleDataSaver(BaseDataSaver):
    """Dataset Saver for simple datas"""

    def save_datas(self, datas: Dict[str, np.ndarray], index: int = -1) -> Dict[str, str]:
        """Save 1 simple datas.

        Parameters
        -------
        datas: dict<str, np.ndarray>
            The datas to be saved.
        indec: int
            The current index

        Returns
        -------
        datas_path: dict<str, str>
           The data paths.
        """

        datas_path = {}
        current = self._current if index < 0 else index
        for name, data in datas.items():
            datas_path[name] = self._save_data(current, name, data, "datas")
        if index > 0:
            self._current = index
        else:
            self._current += 1
        return datas_path

    def setup(self, options: dict):
        return {"datas": {}, "num_datas": 0}


class IODataSaver(BaseDataSaver):
    """Dataset Saver for inputs/outputs"""

    def setup(self, options: dict):
        assert "input_names" in options, "input_names should be given to setup IODataSaver"
        self._input_names = options["input_names"]
        self._output_names = options.get("output_names", [])
        return {
            "inputs": {},
            "outputs": {},
            "num_datas": 0,
            "input_names": self._input_names,
            "output_names": self._output_names,
        }

    def finalize(self):
        """Finalize the saver"""

        super().finalize()
        if any(n not in self._info["inputs"] for n in self._input_names):
            return
        with open(os.path.join(self._folder, "datas_info.txt"), "w") as f:
            for name in self._input_names:
                info = self._info["inputs"][name]
                f.write("{} {} {}\n".format(name, info.get("save_name", name), info["bytes"]))
            for name in self._output_names:
                if name not in self._info["outputs"]:
                    continue
                info = self._info["outputs"][name]
                f.write("{} {} {}\n".format(name, info.get("save_name", name), info["bytes"]))

    def is_finalized(self) -> bool:
        """Check if the saver is finalized

        Returns
        -------
        is_finalized: bool
           Whether the saver is finalized.
        """

        if not super().is_finalized():
            return False
        return os.path.isfile(os.path.join(self._folder, "datas_info.txt"))

    def save_batch(
        self,
        inputs: Union[Dict[str, np.ndarray], List[np.ndarray]],
        outputs: Union[Dict[str, np.ndarray], List[np.ndarray]] = None,
    ) -> int:
        """Save 1 batch inputs and outputs.

        Parameters
        -------
        inputs: list<np.ndarray>/dict<str, np.ndarray>
            The inputs datas.
        outputs: list<np.ndarray>/dict<str, np.ndarray>
            The outputs datas.

        Returns
        -------
        current: int
           The current batch cnt.
        """

        inputs = format_datas(inputs, self._input_names, style="dict")
        for name, data in inputs.items():
            self._save_data(self._current, name, data, "inputs")
        if outputs is not None:
            outputs = format_datas(outputs, self._output_names, style="dict")
            for name, data in outputs.items():
                self._save_data(self._current, name, data, "outputs")
        self._current += 1
        return self._current


def is_io_dataset(folder: str) -> bool:
    """Check if a folder is IO dataset"""

    if not isinstance(folder, str):
        return False
    if not os.path.isfile(os.path.join(folder, "datas_info.json")):
        return False
    data_info = load_dict(os.path.join(folder, "datas_info.json"))
    if any(key not in data_info for key in ["inputs", "outputs", "num_datas"]):
        return False
    return data_info["num_datas"] > 0


def is_simple_dataset(folder: str) -> bool:
    """Check if a folder is simple dataset"""

    if not os.path.isfile(os.path.join(folder, "datas_info.json")):
        return False
    data_info = load_dict(os.path.join(folder, "datas_info.json"))
    if any(key not in data_info for key in ["datas", "num_datas"]):
        return False
    return data_info["num_datas"] > 0
