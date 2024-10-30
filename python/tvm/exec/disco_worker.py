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
# pylint: disable=invalid-name
"""Internal DiscoWorker for Disco ProcessSession."""
import os
import sys

from typing import Callable

import tvm
from tvm._ffi import get_global_func, register_func
from tvm.runtime import NDArray, ShapeTuple, String
from tvm.runtime.ndarray import array


@register_func("tests.disco.add_one", override=True)
def _add_one(x: int) -> int:
    return x + 1


@register_func("tests.disco.add_one_float", override=True)
def _add_one_float(x: float):
    return x + 0.5


@register_func("tests.disco.add_one_ndarray", override=True)
def _add_one_ndarray(x: NDArray) -> NDArray:
    return array(x.numpy() + 1)


@register_func("tests.disco.str", override=True)
def _str_func(x: str):
    return x + "_suffix"


@register_func("tests.disco.str_obj", override=True)
def _str_obj_func(x: String):
    assert isinstance(x, String)
    return String(x + "_suffix")


@register_func("tests.disco.shape_tuple", override=True)
def _shape_tuple_func(x: ShapeTuple):
    assert isinstance(x, ShapeTuple)
    return ShapeTuple(list(x) + [4, 5])


@register_func("tests.disco.test_callback", override=True)
def _make_callback(device: tvm.runtime.Device) -> Callable[[str, int], NDArray]:
    """For use in tests/python/disco/test_callback.py

    This function simulates a callback to be used for lazy parameter
    loading.

    Parameters
    ----------
    device: tvm.runtime.Device

        The device on which parameters should be located, when
        returned by the callback function.

    Returns
    -------
    fget_item: Callable[[str,int], NDArray]

        A callback function that accepts a parameter's name and index,
        and returns the specified parameter.

    """
    import numpy as np  # pylint: disable=import-outside-toplevel

    def fget_item(param_name: str, param_index: int) -> NDArray:
        if param_index == 0:
            assert param_name == "A"
            arr = np.arange(16).reshape([4, 4]).astype("int32")
        elif param_index == 1:
            assert param_name == "B"
            arr = np.arange(4).reshape([2, 2]).astype("float32")
        else:
            raise ValueError(f"Unexpected index {param_index}")
        return tvm.nd.array(arr, device=device)

    return fget_item


def main():
    """Main worker function"""
    if len(sys.argv) != 6:
        print("Usage: <worker_id> <num_workers> <num_groups> <read_fd> <write_fd>")
        return
    worker_id = int(sys.argv[1])
    num_workers = int(sys.argv[2])
    num_groups = int(sys.argv[3])
    if sys.platform == "win32":
        import msvcrt  # pylint: disable=import-outside-toplevel,import-error

        reader = msvcrt.open_osfhandle(int(sys.argv[4]), os.O_BINARY)
        writer = msvcrt.open_osfhandle(int(sys.argv[5]), os.O_BINARY)
    else:
        reader = int(sys.argv[4])
        writer = int(sys.argv[5])

    worker_func = get_global_func("runtime.disco.WorkerProcess")
    worker_func(worker_id, num_workers, num_groups, reader, writer)


if __name__ == "__main__":
    try:
        main()
    except (KeyboardInterrupt, IOError):
        pass
