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
"""tvm.contrib.msc.core.utils.dataset"""

import os
import json
from typing import List
import numpy as np

from .info import load_dict


class MSCDataLoader(object):
    """Dataset Loader for MSC

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
        super(MSCDataLoader, self).__init__()
        self._folder = folder
        self._start = start
        self._current = 0
        assert os.path.isdir(folder), "Dataset {} is not folder".format(folder)
        self._info = load_dict(os.path.join(folder, "msc_info.json"))
        if end == -1:
            self._end = self._info["num_datas"]
        else:
            self._end = min(end, self._info["num_datas"])

    def __getitem__(self, idx):
        if idx + self._start >= self._end:
            raise StopIteration("Reach End")
        if "inputs" in self._info:
            inputs = {n: self._load_data(n, idx, i) for n, i in self._info["inputs"].items()}
        else:
            inputs = {}
        if "outputs" in self._info:
            outputs = {n: self._load_data(n, idx, i) for n, i in self._info["outputs"].items()}
        else:
            outputs = {}
        return inputs, outputs

    def __next__(self):
        if self._current + self._start >= self._end:
            raise StopIteration("Reach End")
        inputs, outputs = self.__getitem__(self._current)
        self._current += 1
        return inputs, outputs

    def __len__(self):
        return self._end - self._start

    def reset(self):
        self._current = 0

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

        f_path = os.path.join(self._folder, name, "batch_{}.bin".format(self._start + index))
        assert os.path.isfile(f_path), "Can not find data file " + str(f_path)
        return np.fromfile(f_path, dtype=info["dtype"]).reshape(info["shape"])


class MSCDataSaver(object):
    """Dataset Saver for MSC

    Parameters
    ----------
    folder: string
        The dataset folder path.
    input_names: list<string>
        The input names.
    output_names: list<string>
        The output names.
    start: int
        The start position.
    max_size: int
        The max size for datas.
    """

    def __init__(
        self,
        folder: str,
        input_names: List[str],
        output_names: List[str],
        start: int = 0,
        max_size: int = -1,
    ):
        super(MSCDataSaver, self).__init__()
        if not os.path.isdir(folder):
            os.mkdir(folder)
        self._folder = folder
        self._input_names = input_names
        self._output_names = output_names
        self._start = start
        self._max_size = max_size
        self._current = 0
        assert os.path.isdir(folder), "Dataset {} is not folder".format(folder)
        self._info = {"inputs": {}, "outputs": {}, "num_datas": 0}

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self._info["num_datas"] = self._current
        with open(os.path.join(self._folder, "msc_info.json"), "w") as f:
            f.write(json.dumps(self._info, indent=2))

    def reset(self):
        self._current = 0

    def save(self, inputs: List[np.ndarray], outputs: List[np.ndarray] = None):
        """Save 1 batch inputs and outputs.

        Parameters
        -------
        inputs: list<np.ndarray>
            The inputs datas.
        outputs: list<np.ndarray>
            The outputs datas.
        """

        assert len(inputs) == len(
            self._input_names
        ), "inputs size {} mismatch with input_names {}".format(len(inputs), self._input_names)
        for idx, i_data in enumerate(inputs):
            self._save_data(self._input_names[idx], i_data, True)
        if outputs:
            assert len(outputs) == len(
                self._output_names
            ), "outputs size {} mismatch with output_names {}".format(
                len(outputs), self._output_names
            )
            for idx, o_data in enumerate(outputs):
                self._save_data(self._output_names[idx], o_data, False)
        self._current += 1

    def _save_data(self, name: str, data: np.ndarray, is_input: bool):
        """Save data to file.

        Parameters
        -------
        name: str
            The name of the data.
        data: np.ndarray
           The data to be saved.
        is_input: bool
            Whether the data is input.
        """

        sub_folder = f_path = os.path.join(self._folder, name)
        if not os.path.isdir(sub_folder):
            os.mkdir(sub_folder)
        f_path = os.path.join(sub_folder, "batch_{}.bin".format(self._start + self._current))
        ref_info = self._info["inputs"] if is_input else self._info["outputs"]
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
            }
        data.tofile(f_path)
